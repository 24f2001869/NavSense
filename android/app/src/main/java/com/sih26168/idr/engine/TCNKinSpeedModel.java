package com.sih26168.idr.engine;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * TCN-kin Speed Model Runner for Android.
 * Ingests the 9-channel kinematic triad:
 *   [a_x, a_y, a_z, omega_yaw, omega_pitch, omega_roll, a_horiz, a_vert, kappa]
 * Maintains a 100-sample (10.0-second) sliding window buffer and executes 1 Hz ONNX
 * inference using Microsoft ONNX Runtime on Android hardware.
 */
public class TCNKinSpeedModel {

    public static final int WINDOW_SIZE = 100;
    public static final int NUM_FEATURES = 9;

    private final float[] mean = new float[NUM_FEATURES];
    private final float[] scale = new float[NUM_FEATURES];

    private final float[][] ringBuffer = new float[WINDOW_SIZE][NUM_FEATURES];
    private int bufferCount = 0;
    private int writeIdx = 0;

    private Object ortEnv = null;
    private Object ortSession = null;
    private boolean onnxInitialized = false;

    private double lastPredictedSpeed = 0.0;
    private double lastInferenceLatencyMs = 0.0;

    public TCNKinSpeedModel(InputStream onnxStream, InputStream scalerStream) {
        loadScaler(scalerStream);
        initOnnx(onnxStream);
    }

    public TCNKinSpeedModel(byte[] onnxBytes, String scalerJsonStr) {
        parseScalerJson(scalerJsonStr);
        initOnnxFromBytes(onnxBytes);
    }

    private void loadScaler(InputStream scalerStream) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[1024];
            int r;
            while ((r = scalerStream.read(buf)) != -1) {
                baos.write(buf, 0, r);
            }
            String jsonStr = baos.toString(StandardCharsets.UTF_8.name());
            parseScalerJson(jsonStr);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    private void parseScalerJson(String jsonStr) {
        try {
            Map<String, Object> map = MiniJson.parseObject(jsonStr);
            List<Object> meanList = (List<Object>) map.get("mean");
            List<Object> scaleList = (List<Object>) map.get("scale");

            for (int i = 0; i < NUM_FEATURES; i++) {
                if (meanList != null && i < meanList.size()) {
                    mean[i] = ((Number) meanList.get(i)).floatValue();
                } else {
                    mean[i] = 0.0f;
                }
                if (scaleList != null && i < scaleList.size()) {
                    scale[i] = ((Number) scaleList.get(i)).floatValue();
                    if (scale[i] == 0.0f) scale[i] = 1.0f;
                } else {
                    scale[i] = 1.0f;
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initOnnx(InputStream onnxStream) {
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int r;
            while ((r = onnxStream.read(buf)) != -1) {
                baos.write(buf, 0, r);
            }
            initOnnxFromBytes(baos.toByteArray());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initOnnxFromBytes(byte[] modelBytes) {
        try {
            Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
            ortEnv = envClass.getMethod("getEnvironment").invoke(null);

            Class<?> sessionClass = Class.forName("ai.onnxruntime.OrtSession");
            ortSession = envClass.getMethod("createSession", byte[].class).invoke(ortEnv, (Object) modelBytes);
            onnxInitialized = true;
        } catch (Throwable t) {
            // Running on desktop JVM without Android ONNX runtime native libraries
            onnxInitialized = false;
        }
    }

    /**
     * Ingests a single 10 Hz IMU sample and updates the 9-channel kinematic feature buffer.
     */
    public void addSample(
        double linAccX, double linAccY, double linAccZ,
        double gyroYaw, double gyroPitch, double gyroRoll,
        double gravX, double gravY, double gravZ
    ) {
        double gNorm = Math.sqrt(gravX * gravX + gravY * gravY + gravZ * gravZ);
        if (gNorm < 1e-6) gNorm = 9.80665;
        double ugx = gravX / gNorm;
        double ugy = gravY / gNorm;
        double ugz = gravZ / gNorm;

        // Kinematic decomposition
        double aVert = linAccX * ugx + linAccY * ugy + linAccZ * ugz;
        double aTotSq = linAccX * linAccX + linAccY * linAccY + linAccZ * linAccZ;
        double aHoriz = Math.sqrt(Math.max(0.0, aTotSq - aVert * aVert));
        double omegaMag = Math.sqrt(gyroYaw * gyroYaw + gyroPitch * gyroPitch + gyroRoll * gyroRoll);
        double kappa = omegaMag * aHoriz;

        ringBuffer[writeIdx][0] = (float) linAccX;
        ringBuffer[writeIdx][1] = (float) linAccY;
        ringBuffer[writeIdx][2] = (float) linAccZ;
        ringBuffer[writeIdx][3] = (float) gyroYaw;
        ringBuffer[writeIdx][4] = (float) gyroPitch;
        ringBuffer[writeIdx][5] = (float) gyroRoll;
        ringBuffer[writeIdx][6] = (float) aHoriz;
        ringBuffer[writeIdx][7] = (float) aVert;
        ringBuffer[writeIdx][8] = (float) kappa;

        writeIdx = (writeIdx + 1) % WINDOW_SIZE;
        if (bufferCount < WINDOW_SIZE) {
            bufferCount++;
        }
    }

    /**
     * Executes 1 Hz forward inference over the current 100-sample temporal window.
     * Returns forward vehicle speed in m/s (guaranteed >= 0.0).
     */
    public double predictSpeed() {
        if (bufferCount < WINDOW_SIZE) {
            return 0.0;
        }

        long tStart = System.nanoTime();

        // Reconstruct chronological window [100, 9]
        float[] flatData = new float[WINDOW_SIZE * NUM_FEATURES];
        int startIdx = (bufferCount == WINDOW_SIZE) ? writeIdx : 0;

        for (int i = 0; i < WINDOW_SIZE; i++) {
            int cur = (startIdx + i) % WINDOW_SIZE;
            for (int c = 0; c < NUM_FEATURES; c++) {
                float val = ringBuffer[cur][c];
                flatData[i * NUM_FEATURES + c] = (val - mean[c]) / scale[c];
            }
        }

        double predSpeed = lastPredictedSpeed;

        if (onnxInitialized && ortSession != null) {
            try {
                Class<?> tensorClass = Class.forName("ai.onnxruntime.OnnxTensor");
                Class<?> envClass = Class.forName("ai.onnxruntime.OrtEnvironment");
                Class<?> sessionClass = Class.forName("ai.onnxruntime.OrtSession");

                FloatBuffer fb = FloatBuffer.wrap(flatData);
                long[] shape = new long[]{1, WINDOW_SIZE, NUM_FEATURES};

                Object tensor = tensorClass.getMethod("createTensor", envClass, FloatBuffer.class, long[].class)
                    .invoke(null, ortEnv, fb, shape);

                String inputName = "input_features";
                Map<String, Object> inputs = Collections.singletonMap(inputName, tensor);

                Object result = sessionClass.getMethod("run", Map.class).invoke(ortSession, inputs);

                // Extract output float array
                Class<?> resultClass = Class.forName("ai.onnxruntime.OrtSession$Result");
                Object outputValue = resultClass.getMethod("get", int.class).invoke(result, 0);
                Class<?> onnxValueClass = Class.forName("ai.onnxruntime.OnnxValue");
                Object valueObj = onnxValueClass.getMethod("getValue").invoke(outputValue);

                if (valueObj instanceof float[][]) {
                    predSpeed = ((float[][]) valueObj)[0][0];
                } else if (valueObj instanceof float[]) {
                    predSpeed = ((float[]) valueObj)[0];
                }

                predSpeed = Math.max(0.0, predSpeed); // Physical non-negative constraint
            } catch (Exception e) {
                predSpeed = lastPredictedSpeed;
            }
        }

        long tEnd = System.nanoTime();
        lastInferenceLatencyMs = (tEnd - tStart) / 1e6;
        lastPredictedSpeed = predSpeed;
        return predSpeed;
    }

    public boolean isReady() {
        return bufferCount >= WINDOW_SIZE;
    }

    public double getLastPredictedSpeed() {
        return lastPredictedSpeed;
    }

    public double getLastInferenceLatencyMs() {
        return lastInferenceLatencyMs;
    }

    public boolean isOnnxInitialized() {
        return onnxInitialized;
    }
}
