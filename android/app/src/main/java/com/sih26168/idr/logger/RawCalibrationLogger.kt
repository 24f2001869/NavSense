package com.sih26168.idr.logger

import android.content.Context
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import com.sih26168.idr.engine.Vector3
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * High-precision, zero-preprocessing calibration data logger for Stage C11.
 * 
 * Captures raw sensor streams, device attitude representations, and high-accuracy GNSS
 * reference fields under a single synchronized monotonic hardware timebase (SystemClock.elapsedRealtimeNanos).
 * 
 * Runs all disk I/O on a dedicated background HandlerThread to prevent UI stutter and guarantee
 * zero dropped samples up to 100+ Hz.
 */
class RawCalibrationLogger(private val context: Context) {

    companion object {
        private const val TAG = "RawCalibrationLogger"
        
        const val CSV_HEADER = "timestamp_ns,utc_time_ms," +
                "accel_x,accel_y,accel_z," +
                "gyro_x,gyro_y,gyro_z," +
                "mag_x,mag_y,mag_z," +
                "gravity_x,gravity_y,gravity_z," +
                "lin_accel_x,lin_accel_y,lin_accel_z," +
                "rot_vec_x,rot_vec_y,rot_vec_z,rot_vec_w," +
                "gnss_lat,gnss_lon,gnss_alt,gnss_speed,gnss_bearing," +
                "gnss_acc_h,gnss_acc_v,gnss_acc_speed,gnss_acc_bearing," +
                "gnss_elapsed_ns,gnss_age_ms,protocol_marker"
    }

    private var ioThread: HandlerThread? = null
    private var ioHandler: Handler? = null

    private var writer: PrintWriter? = null
    private var currentFile: File? = null

    @Volatile
    var isLogging: Boolean = false
        private set

    @Volatile
    var activeProtocolMarker: String = "NORMAL"
        private set

    @Volatile
    var sampleCount: Long = 0L
        private set

    fun setProtocolMarker(marker: String) {
        activeProtocolMarker = marker
        Log.i(TAG, "Active Protocol Marker updated to: $marker")
    }

    fun startLogging(customTag: String = "calibration"): File? {
        if (isLogging) return currentFile

        return try {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(dir, "raw_${customTag}_$timeStamp.csv")

            val thread = HandlerThread("RawLoggerIOThread").apply { start() }
            ioThread = thread
            val handler = Handler(thread.looper)
            ioHandler = handler

            writer = PrintWriter(BufferedWriter(FileWriter(file), 65536))
            writer?.println(CSV_HEADER)
            writer?.flush()

            currentFile = file
            sampleCount = 0L
            isLogging = true

            Log.i(TAG, "Started raw calibration logging to: ${file.absolutePath}")
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start raw calibration logging", e)
            null
        }
    }

    /**
     * Non-blocking sample ingestion. Formats row and enqueues to background I/O handler.
     */
    fun logSample(
        timestampNs: Long,
        utcTimeMs: Long,
        accel: Vector3,
        gyro: Vector3,
        mag: Vector3,
        gravity: Vector3,
        linAccel: Vector3,
        rotVec: FloatArray, // [x, y, z, w]
        gnssLat: Double,
        gnssLon: Double,
        gnssAlt: Double,
        gnssSpeed: Double,
        gnssBearing: Double,
        gnssAccH: Double,
        gnssAccV: Double,
        gnssAccSpeed: Double,
        gnssAccBearing: Double,
        gnssElapsedNs: Long,
        currentMarker: String = activeProtocolMarker
    ) {
        if (!isLogging) return

        val handler = ioHandler ?: return
        val nowNs = SystemClock.elapsedRealtimeNanos()
        val gnssAgeMs = if (gnssElapsedNs > 0) ((nowNs - gnssElapsedNs) / 1_000_000L).coerceAtLeast(0L) else -1L

        val rx = if (rotVec.isNotEmpty()) rotVec[0] else 0f
        val ry = if (rotVec.size > 1) rotVec[1] else 0f
        val rz = if (rotVec.size > 2) rotVec[2] else 0f
        val rw = if (rotVec.size > 3) rotVec[3] else 1f

        handler.post {
            try {
                writer?.let { w ->
                    val line = String.format(
                        Locale.US,
                        "%d,%d," +
                                "%.5f,%.5f,%.5f," +
                                "%.6f,%.6f,%.6f," +
                                "%.2f,%.2f,%.2f," +
                                "%.5f,%.5f,%.5f," +
                                "%.5f,%.5f,%.5f," +
                                "%.6f,%.6f,%.6f,%.6f," +
                                "%.8f,%.8f,%.2f,%.3f,%.2f," +
                                "%.2f,%.2f,%.2f,%.2f," +
                                "%d,%d,%s",
                        timestampNs, utcTimeMs,
                        accel.x, accel.y, accel.z,
                        gyro.x, gyro.y, gyro.z,
                        mag.x, mag.y, mag.z,
                        gravity.x, gravity.y, gravity.z,
                        linAccel.x, linAccel.y, linAccel.z,
                        rx, ry, rz, rw,
                        gnssLat, gnssLon, gnssAlt, gnssSpeed, gnssBearing,
                        gnssAccH, gnssAccV, gnssAccSpeed, gnssAccBearing,
                        gnssElapsedNs, gnssAgeMs, currentMarker
                    )
                    w.println(line)
                    sampleCount++
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error writing raw sample", e)
            }
        }
    }

    fun stopLogging(): File? {
        if (!isLogging) return null

        val file = currentFile
        isLogging = false

        ioHandler?.post {
            try {
                writer?.flush()
                writer?.close()
                writer = null
                Log.i(TAG, "Raw calibration log cleanly flushed and closed: ${file?.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "Error closing raw calibration log", e)
            } finally {
                ioThread?.quitSafely()
                ioThread = null
                ioHandler = null
            }
        }

        currentFile = null
        return file
    }
}
