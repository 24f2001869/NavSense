package com.sih26168.idr

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sih26168.idr.engine.*
import com.sih26168.idr.logger.RawCalibrationLogger
import com.sih26168.idr.logger.SensorDataLogger
import com.sih26168.idr.ui.AttitudeHorizonView
import com.sih26168.idr.ui.CompassRoseView
import com.sih26168.idr.ui.LocalRoadMapView
import java.io.File
import java.io.InputStream
import java.nio.charset.StandardCharsets
import java.util.*

/**
 * Main Automotive HUD Activity: Live Field Diagnostics + Navigation Observer.
 * Bridges smartphone IMU/GNSS hardware into the unified 15-state DeadReckoningEngine.
 * Provides transparent physical verification before Stage C10.2 Test 3.
 *
 * Supports:
 * - Mode A: Interactive 60-second GNSS blackout replay demo (Golden Reference session).
 * - Mode B: Real smartphone sensor pipeline with live 15-state ESKF,
 *           Test 1 CSV telemetry logging, and Test 2 GNSS blackout simulation toggle.
 */
class MainActivity : AppCompatActivity(), SensorEventListener, LocationListener {

    // 1. Header & Navigation State
    private lateinit var badgeNavState: TextView
    private lateinit var tvTripTitle: TextView

    // 2. Local OSM Road Map View
    private lateinit var roadMapView: LocalRoadMapView
    private lateinit var tvGnssLatLon: TextView
    private lateinit var tvGnssAccuracy: TextView

    // 3. Side-by-Side Speed Comparison
    private lateinit var tvGnssSpeedKmh: TextView
    private lateinit var tvGnssSpeedMs: TextView
    private lateinit var tvSpeedKmh: TextView
    private lateinit var tvSpeedMs: TextView
    private lateinit var tvSpeedError: TextView

    // 4. Heading & Compass Display
    private lateinit var compassView: CompassRoseView
    private lateinit var tvHeadingDeg: TextView
    private lateinit var tvHeadingCardinal: TextView
    private lateinit var tvHeadingSource: TextView
    private lateinit var tvGnssCourse: TextView
    private lateinit var tvFilterHeading: TextView
    private lateinit var tvHeadingDelta: TextView
    private lateinit var tvMagHeading: TextView

    // 5. Phone Tilt / Attitude & Altitude / Vertical Motion
    private lateinit var horizonView: AttitudeHorizonView
    private lateinit var tvPhoneAttitude: TextView
    private lateinit var tvGnssAlt: TextView
    private lateinit var tvFilterUp: TextView
    private lateinit var tvVertSpeed: TextView
    private lateinit var tvVerticalDriftWarn: TextView

    // 6. Sensor Health & Timing
    private lateinit var tileSensorAccel: TextView
    private lateinit var tileSensorGyro: TextView
    private lateinit var tileSensorMag: TextView
    private lateinit var tileSensorGnss: TextView
    private lateinit var tvNormAccel: TextView
    private lateinit var tvNormGyro: TextView
    private lateinit var tvNormMag: TextView
    private lateinit var tvSensorRate: TextView

    // 7. Constraints & Quality Gates
    private lateinit var tileNhc: TextView
    private lateinit var tileVnhc: TextView
    private lateinit var tileCompass: TextView
    private lateinit var tileOsm: TextView
    private lateinit var tvBiasAccel: TextView
    private lateinit var tvBiasGyro: TextView

    // 8. Local Cartesian ENU Position & Outage Timer
    private lateinit var tvPosSigma: TextView
    private lateinit var tvPosE: TextView
    private lateinit var tvPosN: TextView
    private lateinit var tvPosU: TextView
    private lateinit var tvOutageTimer: TextView

    // 9. Buttons & Readouts
    private lateinit var btnReplayDemo: Button
    private lateinit var btnRecordLog: Button
    private lateinit var btnToggleOutage: Button
    private lateinit var tvLogStatus: TextView

    // 10. Stage C11 Controlled Calibration Recorder
    private lateinit var rawCalibrationLogger: RawCalibrationLogger
    private lateinit var btnRecordRawCalibration: Button
    private lateinit var tvRawLogStatus: TextView
    private lateinit var tvActiveMarkerBadge: TextView
    private val protocolButtons = mutableListOf<Button>()

    // Engine, Models & Logger
    private lateinit var sensorManager: SensorManager
    private var locationManager: LocationManager? = null
    private var engine: DeadReckoningEngine? = null
    private var roadIndex: RoadNetworkIndex? = null
    private val speedModel = CausalSpeedModel()
    private var tcnSpeedModel: TCNKinSpeedModel? = null
    private val liveFeatureExtractor = LiveFeatureExtractor(10)
    private lateinit var sensorLogger: SensorDataLogger

    // Live sensor buffers
    private val curAccel = Vector3()
    private val curGyro = Vector3()
    private val curMag = Vector3()
    private val curGravity = Vector3()
    private val curLinAccel = Vector3()
    private val curRotVec = FloatArray(4)
    private var lastMag = Vector3()
    private var lastMagTime = 0L

    // Real-Time Tilt-Compensated Heading & Attitude Tracking
    private var isHeadingInitialized = false
    private var livePsiMagDeg = 0.0
    private var isMagValid = false
    private var livePitchDeg = 0.0f
    private var liveRollDeg = 0.0f

    // Live GNSS buffer & geodetic origin
    private var lastLocation: Location? = null
    private var originLat = 0.0
    private var originLon = 0.0
    private var originAlt = 0.0
    private var hasOrigin = false
    private var isOutageSimulated = false

    // Loop Timing (measured from actual timestamps, not assumed 10Hz)
    private var lastLoopTimeNs = 0L
    private var measuredHz = 10.0
    private var measuredDtMs = 100.0

    // Live execution loop
    private val liveLoopHandler = Handler(Looper.getMainLooper())
    private var loggedEpochCount = 0
    private val liveLoopRunnable = object : Runnable {
        override fun run() {
            if (!isReplayRunning) {
                stepLiveEngine()
            }
            liveLoopHandler.postDelayed(this, 100L) // nominal 10 Hz trigger
        }
    }

    // Demo Replay state
    private var isReplayRunning = false
    private var replayThread: Thread? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        sensorLogger = SensorDataLogger(this)
        rawCalibrationLogger = RawCalibrationLogger(this)

        bindViews()
        initEngineFromAssets()
        initSensors()
        initLocation()

        btnReplayDemo.setOnClickListener {
            if (isReplayRunning) {
                stopReplayDemo()
            } else {
                startReplayDemo()
            }
        }

        btnRecordLog.setOnClickListener {
            toggleLogging()
        }

        btnToggleOutage.setOnClickListener {
            toggleOutageSimulation()
        }

        btnRecordRawCalibration.setOnClickListener {
            toggleRawCalibrationLogging()
        }

        setupProtocolTagButtons()

        // Start live execution loop
        liveLoopHandler.postDelayed(liveLoopRunnable, 500L)
    }

    private fun bindViews() {
        // 1. Header & State
        badgeNavState = findViewById(R.id.badge_nav_state)
        tvTripTitle = findViewById(R.id.tv_trip_title)

        // 2. Map
        roadMapView = findViewById(R.id.road_map_view)
        tvGnssLatLon = findViewById(R.id.tv_gnss_lat_lon)
        tvGnssAccuracy = findViewById(R.id.tv_gnss_accuracy)

        // 3. Speed Comparison
        tvGnssSpeedKmh = findViewById(R.id.tv_gnss_speed_kmh)
        tvGnssSpeedMs = findViewById(R.id.tv_gnss_speed_ms)
        tvSpeedKmh = findViewById(R.id.tv_speed_kmh)
        tvSpeedMs = findViewById(R.id.tv_speed_ms)
        tvSpeedError = findViewById(R.id.tv_speed_error)

        // 4. Heading & Compass
        compassView = findViewById(R.id.compass_view)
        tvHeadingDeg = findViewById(R.id.tv_heading_deg)
        tvHeadingCardinal = findViewById(R.id.tv_heading_cardinal)
        tvHeadingSource = findViewById(R.id.tv_heading_source)
        tvGnssCourse = findViewById(R.id.tv_gnss_course)
        tvFilterHeading = findViewById(R.id.tv_filter_heading)
        tvHeadingDelta = findViewById(R.id.tv_heading_delta)
        tvMagHeading = findViewById(R.id.tv_mag_heading)

        // 5. Attitude & Altitude
        horizonView = findViewById(R.id.horizon_view)
        tvPhoneAttitude = findViewById(R.id.tv_phone_attitude)
        tvGnssAlt = findViewById(R.id.tv_gnss_alt)
        tvFilterUp = findViewById(R.id.tv_filter_up)
        tvVertSpeed = findViewById(R.id.tv_vert_speed)
        tvVerticalDriftWarn = findViewById(R.id.tv_vertical_drift_warn)

        // 6. Sensor Health & Timing
        tileSensorAccel = findViewById(R.id.tile_sensor_accel)
        tileSensorGyro = findViewById(R.id.tile_sensor_gyro)
        tileSensorMag = findViewById(R.id.tile_sensor_mag)
        tileSensorGnss = findViewById(R.id.tile_sensor_gnss)
        tvNormAccel = findViewById(R.id.tv_norm_accel)
        tvNormGyro = findViewById(R.id.tv_norm_gyro)
        tvNormMag = findViewById(R.id.tv_norm_mag)
        tvSensorRate = findViewById(R.id.tv_sensor_rate)

        // 7. Constraints
        tileNhc = findViewById(R.id.tile_nhc)
        tileVnhc = findViewById(R.id.tile_vnhc)
        tileCompass = findViewById(R.id.tile_compass)
        tileOsm = findViewById(R.id.tile_osm)
        tvBiasAccel = findViewById(R.id.tv_bias_accel)
        tvBiasGyro = findViewById(R.id.tv_bias_gyro)

        // 8. Position & Outage Timer
        tvPosSigma = findViewById(R.id.tv_pos_sigma)
        tvPosE = findViewById(R.id.tv_pos_e)
        tvPosN = findViewById(R.id.tv_pos_n)
        tvPosU = findViewById(R.id.tv_pos_u)
        tvOutageTimer = findViewById(R.id.tv_outage_timer)

        // 9. Buttons
        btnReplayDemo = findViewById(R.id.btn_replay_demo)
        btnRecordLog = findViewById(R.id.btn_record_log)
        btnToggleOutage = findViewById(R.id.btn_toggle_outage)
        tvLogStatus = findViewById(R.id.tv_log_status)

        // Stage C11 Controlled Calibration Recorder
        btnRecordRawCalibration = findViewById(R.id.btn_record_raw_calibration)
        tvRawLogStatus = findViewById(R.id.tv_raw_log_status)
        tvActiveMarkerBadge = findViewById(R.id.tv_active_marker_badge)
    }

    private fun initEngineFromAssets() {
        try {
            // Load OSM road network from assets
            val roadStream: InputStream = assets.open("vta04_road_network.json")
            val bytes = ByteArray(roadStream.available())
            roadStream.read(bytes)
            roadStream.close()
            val roadJson = String(bytes, StandardCharsets.UTF_8)
            val roadMap = MiniJson.parseObject(roadJson)
            @Suppress("UNCHECKED_CAST")
            val segListJson = roadMap["segments"] as List<Map<String, Any>>

            val segments = ArrayList<RoadNetworkIndex.Segment>()
            for (s in segListJson) {
                @Suppress("UNCHECKED_CAST")
                val p1 = s["p1"] as List<Number>
                @Suppress("UNCHECKED_CAST")
                val p2 = s["p2"] as List<Number>
                val headingRad = (s["heading_rad"] as Number).toDouble()
                val wayId = (s["way_id"] as Number).toLong()
                segments.add(RoadNetworkIndex.Segment(
                    p1[0].toDouble(), p1[1].toDouble(),
                    p2[0].toDouble(), p2[1].toDouble(),
                    headingRad, wayId
                ))
            }
            roadIndex = RoadNetworkIndex(segments)
            roadMapView.setRoadNetwork(segments)

            val cfg = DeadReckoningEngine.EngineConfig().apply {
                sigma_speed = 0.60
                sigma_lat_0 = 0.50
                mag_norm_tol = 0.08
                mag_db_dt_tol = 5.0
                enable_vert_nhc = true
                turn_rate_vnhc_gate_deg_s = 3.0
                turn_rate_compass_gate_deg_s = 3.0
            }

            try {
                val onnxStream = assets.open("tcn_kin_speed.onnx")
                val scalerStream = assets.open("tcn_scaler.json")
                tcnSpeedModel = TCNKinSpeedModel(onnxStream, scalerStream)
                android.util.Log.i("MainActivity", "TCN-kin ONNX engine successfully initialized!")
            } catch (e: Exception) {
                android.util.Log.e("MainActivity", "TCN ONNX load failed: ${e.message}")
            }

            engine = DeadReckoningEngine(
                cfg,
                roadIndex,
                Vector3(0.0, 0.0, 0.0),
                Vector3(0.0, 0.0, 0.0),
                0.0,
                Matrix.identity(3),
                Vector3(0.0, 0.0, 0.0),
                Vector3(0.0, 0.0, 0.0),
                45.0
            )
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun initSensors() {
        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val accel = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val mag = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
        val grav = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY)
        val linAccel = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION)
        val rotVec = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

        accel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        gyro?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        mag?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        grav?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        linAccel?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
        rotVec?.let { sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_FASTEST) }
    }

    private fun initLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION), 1001)
        } else {
            startLocationUpdates()
        }
    }

    private fun startLocationUpdates() {
        try {
            locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
            locationManager?.requestLocationUpdates(LocationManager.GPS_PROVIDER, 100L, 0.0f, this)
        } catch (e: SecurityException) {
            e.printStackTrace()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001 && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startLocationUpdates()
        }
    }

    override fun onLocationChanged(location: Location) {
        lastLocation = location
        if (!hasOrigin) {
            originLat = location.latitude
            originLon = location.longitude
            originAlt = location.altitude
            hasOrigin = true
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (isReplayRunning) return // Replay demo takes priority over live sensor events

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                curAccel.x = event.values[0].toDouble()
                curAccel.y = event.values[1].toDouble()
                curAccel.z = event.values[2].toDouble()

                if (rawCalibrationLogger.isLogging) {
                    val loc = lastLocation
                    rawCalibrationLogger.logSample(
                        timestampNs = event.timestamp,
                        utcTimeMs = System.currentTimeMillis(),
                        accel = curAccel,
                        gyro = curGyro,
                        mag = curMag,
                        gravity = curGravity,
                        linAccel = curLinAccel,
                        rotVec = curRotVec,
                        gnssLat = loc?.latitude ?: 0.0,
                        gnssLon = loc?.longitude ?: 0.0,
                        gnssAlt = loc?.altitude ?: 0.0,
                        gnssSpeed = loc?.speed?.toDouble() ?: 0.0,
                        gnssBearing = loc?.bearing?.toDouble() ?: 0.0,
                        gnssAccH = loc?.accuracy?.toDouble() ?: 999.0,
                        gnssAccV = if (loc != null && loc.hasVerticalAccuracy()) loc.verticalAccuracyMeters.toDouble() else 999.0,
                        gnssAccSpeed = if (loc != null && loc.hasSpeedAccuracy()) loc.speedAccuracyMetersPerSecond.toDouble() else 999.0,
                        gnssAccBearing = if (loc != null && loc.hasBearingAccuracy()) loc.bearingAccuracyDegrees.toDouble() else 999.0,
                        gnssElapsedNs = loc?.elapsedRealtimeNanos ?: 0L
                    )
                }
            }
            Sensor.TYPE_GYROSCOPE -> {
                curGyro.x = event.values[0].toDouble()
                curGyro.y = event.values[1].toDouble()
                curGyro.z = event.values[2].toDouble()
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                curMag.x = event.values[0].toDouble()
                curMag.y = event.values[1].toDouble()
                curMag.z = event.values[2].toDouble()
            }
            Sensor.TYPE_GRAVITY -> {
                curGravity.x = event.values[0].toDouble()
                curGravity.y = event.values[1].toDouble()
                curGravity.z = event.values[2].toDouble()
            }
            Sensor.TYPE_LINEAR_ACCELERATION -> {
                curLinAccel.x = event.values[0].toDouble()
                curLinAccel.y = event.values[1].toDouble()
                curLinAccel.z = event.values[2].toDouble()
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                if (event.values.size >= 4) {
                    curRotVec[0] = event.values[0]
                    curRotVec[1] = event.values[1]
                    curRotVec[2] = event.values[2]
                    curRotVec[3] = event.values[3]
                } else if (event.values.size >= 3) {
                    curRotVec[0] = event.values[0]
                    curRotVec[1] = event.values[1]
                    curRotVec[2] = event.values[2]
                    val wSq = 1.0f - (curRotVec[0] * curRotVec[0] + curRotVec[1] * curRotVec[1] + curRotVec[2] * curRotVec[2])
                    curRotVec[3] = if (wSq > 0f) Math.sqrt(wSq.toDouble()).toFloat() else 0f
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    /**
     * Mode B: Periodic execution for real phone sensor processing.
     * Accurately measures actual loop timing instead of assuming 10 Hz.
     */
    private fun stepLiveEngine() {
        val eng = engine ?: return

        val nowMs = System.currentTimeMillis()
        val nowNs = android.os.SystemClock.elapsedRealtimeNanos()

        // Derive dt from monotonic Android timing (rejecting unreasonable anomalies and handling first sample)
        val dt: Double
        if (lastLoopTimeNs > 0L) {
            val rawDt = (nowNs - lastLoopTimeNs) * 1e-9
            if (rawDt in 0.005..1.0) {
                dt = rawDt
                val dtMs = rawDt * 1000.0
                measuredHz = 0.85 * measuredHz + 0.15 * (1.0 / rawDt)
                measuredDtMs = 0.85 * measuredDtMs + 0.15 * dtMs
            } else {
                dt = 0.1 // Safe fallback if process experienced an anomalous pause
            }
        } else {
            dt = 0.1 // Safe initial sample handling
        }
        lastLoopTimeNs = nowNs

        // Magnetic temporal gradient
        var dbDt = 0.0
        if (lastMagTime > 0) {
            val dtMag = (nowMs - lastMagTime) / 1000.0
            if (dtMag > 1e-4) {
                val db = curMag.minus(lastMag).norm()
                dbDt = db / dtMag
            }
        }
        lastMag = Vector3(curMag.x, curMag.y, curMag.z)
        lastMagTime = nowMs

        // 1. Calculate true tilt-compensated geomagnetic azimuth from Rotation Vector or Gravity + Magnetometer
        val rMat = FloatArray(9)
        val orient = FloatArray(3)
        var psiMagDeg = 0.0
        var magValid = false

        if (curRotVec[0] != 0f || curRotVec[1] != 0f || curRotVec[2] != 0f || curRotVec[3] != 0f) {
            SensorManager.getRotationMatrixFromVector(rMat, curRotVec)
            SensorManager.getOrientation(rMat, orient)
            psiMagDeg = (Math.toDegrees(orient[0].toDouble()) + 360.0) % 360.0
            livePitchDeg = Math.toDegrees(orient[1].toDouble()).toFloat()
            liveRollDeg = Math.toDegrees(orient[2].toDouble()).toFloat()
            magValid = true
        } else if (curGravity.norm() > 7.0 && curMag.norm() > 10.0) {
            val gArr = floatArrayOf(curGravity.x.toFloat(), curGravity.y.toFloat(), curGravity.z.toFloat())
            val mArr = floatArrayOf(curMag.x.toFloat(), curMag.y.toFloat(), curMag.z.toFloat())
            val iMat = FloatArray(9)
            if (SensorManager.getRotationMatrix(rMat, iMat, gArr, mArr)) {
                SensorManager.getOrientation(rMat, orient)
                psiMagDeg = (Math.toDegrees(orient[0].toDouble()) + 360.0) % 360.0
                livePitchDeg = Math.toDegrees(orient[1].toDouble()).toFloat()
                liveRollDeg = Math.toDegrees(orient[2].toDouble()).toFloat()
                magValid = true
            }
        }
        livePsiMagDeg = psiMagDeg
        isMagValid = magValid

        // One-time instant initial alignment when valid compass becomes available
        if (!isHeadingInitialized && magValid) {
            eng.setHeadingDeg(psiMagDeg)
            isHeadingInitialized = true
        }

        // Feature extraction for Causal ML Speed & TCN-kin ONNX Model
        val isStationary = curGyro.norm() < 0.05 && Math.abs(curAccel.norm() - 9.81) < 0.3
        val features = liveFeatureExtractor.extractFeatures(
            curAccel, curGyro,
            eng.getPitchDeg(), eng.getRollDeg(),
            psiMagDeg, isStationary, dt
        )
        val mlSpeed = speedModel.predict(features)

        // Feed live IMU triad (linear accel, gyro, gravity) into TCN-kin ONNX inference buffer
        tcnSpeedModel?.addSample(
            curLinAccel.x, curLinAccel.y, curLinAccel.z,
            curGyro.x, curGyro.y, curGyro.z,
            curGravity.x, curGravity.y, curGravity.z
        )
        val tcnPredictedSpeed = tcnSpeedModel?.predictSpeed() ?: -1.0
        val effectiveSpeedEst = if (tcnPredictedSpeed >= 0.0) tcnPredictedSpeed else mlSpeed

        // GNSS measurement construction
        val loc = lastLocation
        val isGnssAvailable = (loc != null && !isOutageSimulated && (nowMs - loc.time < 3000))
        var gnssPosE = 0.0
        var gnssPosN = 0.0
        val gnssMeasurement = if (isGnssAvailable && loc != null) {
            val dLat = Math.toRadians(loc.latitude - originLat)
            val dLon = Math.toRadians(loc.longitude - originLon)
            val latRad = Math.toRadians(originLat)
            gnssPosE = 6378137.0 * dLon * Math.cos(latRad)
            gnssPosN = 6378137.0 * dLat
            val gnssPosU = loc.altitude - originAlt

            val spd = loc.speed.toDouble()
            val bearRad = Math.toRadians(loc.bearing.toDouble())
            val velE = spd * Math.sin(bearRad)
            val velN = spd * Math.cos(bearRad)

            DeadReckoningEngine.GNSSMeasurement(
                true,
                Vector3(gnssPosE, gnssPosN, gnssPosU),
                Vector3(velE, velN, 0.0),
                loc.accuracy.toDouble().coerceAtLeast(2.0)
            )
        } else {
            DeadReckoningEngine.GNSSMeasurement(false)
        }

        // Steer filter heading:
        // Case A: Moving with valid GNSS course (loc.speed >= 1.5 m/s) -> align to road direction
        if (isGnssAvailable && loc != null && loc.speed >= 1.5f && loc.hasBearing()) {
            val gnssCourse = (loc.bearing.toDouble() % 360.0 + 360.0) % 360.0
            eng.alignHeading(gnssCourse, 0.05)
        }
        // Case B: Stationary / bench test -> softly track true magnetic compass
        else if (magValid && !isOutageSimulated && (loc == null || loc.speed < 1.2f)) {
            eng.alignHeading(psiMagDeg, 0.15)
        }
        // Case C: During Outage -> pure gyro dead reckoning + V1 decoupled lateral velocity damping (zero external steering)

        // Step 15-state ESKF with Decoupled Pure-Velocity Damping (V1)
        val tel = eng.step(curAccel, curGyro, effectiveSpeedEst, dt, curMag, gnssMeasurement, psiMagDeg, dbDt)

        // Log epoch if logger is active
        if (sensorLogger.isLogging) {
            loggedEpochCount++
            sensorLogger.logEpoch(
                nowMs,
                dt,
                curAccel, curGyro, curMag, curGravity,
                loc?.latitude ?: 0.0,
                loc?.longitude ?: 0.0,
                loc?.altitude ?: 0.0,
                loc?.speed?.toDouble() ?: 0.0,
                loc?.bearing?.toDouble() ?: 0.0,
                loc?.accuracy?.toDouble() ?: 999.0,
                tel, effectiveSpeedEst
            )
            tvLogStatus.text = "🔴 Recording... $loggedEpochCount epochs logged (${String.format(Locale.US, "%.1f s", loggedEpochCount * dt)})"
        }

        if (rawCalibrationLogger.isLogging) {
            tvRawLogStatus.text = "🔴 Recording... ${rawCalibrationLogger.sampleCount} samples [${rawCalibrationLogger.activeProtocolMarker}]"
        }

        // Update UI with full diagnostics
        updateUi(
            tel, effectiveSpeedEst, isGnssAvailable,
            loc, gnssPosE, gnssPosN, psiMagDeg, dbDt
        )
    }

    private fun toggleLogging() {
        if (sensorLogger.isLogging) {
            val file = sensorLogger.stopLogging()
            btnRecordLog.text = "🔴 START LOGGING"
            btnRecordLog.setTextColor(Color.parseColor("#FF5252"))
            tvLogStatus.text = "Saved ${loggedEpochCount} epochs: ${file?.name} (Ready to pull via adb)"
            Toast.makeText(this, "Log saved to: ${file?.absolutePath}", Toast.LENGTH_LONG).show()
        } else {
            loggedEpochCount = 0
            val file = sensorLogger.startLogging()
            if (file != null) {
                btnRecordLog.text = "⏹ STOP & SAVE LOG"
                btnRecordLog.setTextColor(Color.parseColor("#00E676"))
                tvLogStatus.text = "Recording to: ${file.name}"
                Toast.makeText(this, "Started logging 10Hz telemetry", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun toggleRawCalibrationLogging() {
        if (rawCalibrationLogger.isLogging) {
            val count = rawCalibrationLogger.sampleCount
            val file = rawCalibrationLogger.stopLogging()
            btnRecordRawCalibration.text = "🔴 START RAW CALIBRATION LOG (100 Hz)"
            btnRecordRawCalibration.setTextColor(Color.parseColor("#FF5252"))
            tvRawLogStatus.text = "Saved $count raw samples: ${file?.name} (Ready for ML pipeline)"
            Toast.makeText(this, "Raw log saved to: ${file?.absolutePath}", Toast.LENGTH_LONG).show()
        } else {
            val file = rawCalibrationLogger.startLogging("nord_c11")
            if (file != null) {
                btnRecordRawCalibration.text = "⏹ STOP & SAVE RAW LOG"
                btnRecordRawCalibration.setTextColor(Color.parseColor("#00E676"))
                tvRawLogStatus.text = "Recording raw stream to: ${file.name}"
                Toast.makeText(this, "Started raw calibration logging", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupProtocolTagButtons() {
        protocolButtons.clear()
        val tagMap = listOf(
            Pair(R.id.btn_tag_stat_off, "STAT_OFF"),
            Pair(R.id.btn_tag_stat_on, "STAT_ON"),
            Pair(R.id.btn_tag_cruise_20, "CRUISE_20"),
            Pair(R.id.btn_tag_cruise_40, "CRUISE_40"),
            Pair(R.id.btn_tag_cruise_60, "CRUISE_60"),
            Pair(R.id.btn_tag_accel, "ACCEL"),
            Pair(R.id.btn_tag_brake, "BRAKE"),
            Pair(R.id.btn_tag_turn, "TURN"),
            Pair(R.id.btn_tag_bump, "BUMP"),
            Pair(R.id.btn_tag_normal, "NORMAL")
        )

        for ((id, tag) in tagMap) {
            val btn = findViewById<Button>(id)
            protocolButtons.add(btn)
            btn.setOnClickListener {
                selectProtocolTag(tag, btn)
            }
        }
    }

    private fun selectProtocolTag(marker: String, selectedBtn: Button) {
        rawCalibrationLogger.setProtocolMarker(marker)
        tvActiveMarkerBadge.text = marker
        for (btn in protocolButtons) {
            if (btn == selectedBtn) {
                btn.setTextColor(Color.parseColor("#00E676"))
            } else {
                btn.setTextColor(Color.parseColor("#FFFFFF"))
            }
        }
        Toast.makeText(this, "Protocol Marker: $marker", Toast.LENGTH_SHORT).show()
    }

    private fun toggleOutageSimulation() {
        isOutageSimulated = !isOutageSimulated
        if (isOutageSimulated) {
            btnToggleOutage.text = "🛰 RESTORE GNSS"
            btnToggleOutage.setTextColor(Color.parseColor("#00E676"))
            Toast.makeText(this, "Software-controlled GNSS blackout active (Filter dead-reckoning test)", Toast.LENGTH_SHORT).show()
        } else {
            btnToggleOutage.text = "⚡ SOFTWARE BLACKOUT"
            btnToggleOutage.setTextColor(Color.parseColor("#FFAB00"))
            Toast.makeText(this, "Restoring GNSS Fix (Reacquisition active)", Toast.LENGTH_SHORT).show()
        }
    }

    private fun startReplayDemo() {
        isReplayRunning = true
        btnReplayDemo.text = "⏸  PAUSE REPLAY DEMO"
        btnReplayDemo.setBackgroundColor(Color.parseColor("#FFAB00"))

        replayThread = Thread {
            try {
                val stream: InputStream = assets.open("golden_reference_session.json")
                val bytes = ByteArray(stream.available())
                stream.read(bytes)
                stream.close()
                val sessionJson = String(bytes, StandardCharsets.UTF_8)
                val sessionMap = MiniJson.parseObject(sessionJson)

                @Suppress("UNCHECKED_CAST")
                val meta = sessionMap["meta"] as Map<String, Any>
                @Suppress("UNCHECKED_CAST")
                val epochs = sessionMap["epochs"] as List<Map<String, Any>>

                @Suppress("UNCHECKED_CAST")
                val initPos = meta["init_pos_enu"] as List<Number>
                @Suppress("UNCHECKED_CAST")
                val initVel = meta["init_vel_enu"] as List<Number>
                val initHeading = (meta["init_heading_deg"] as Number).toDouble()
                @Suppress("UNCHECKED_CAST")
                val baStat = meta["ba_stat"] as List<Number>
                val baselineMag = (meta["baseline_mag_uT"] as Number).toDouble()

                engine = DeadReckoningEngine(
                    DeadReckoningEngine.EngineConfig().apply {
                        sigma_speed = 0.60
                        sigma_lat_0 = 0.50
                        mag_norm_tol = 0.08
                        mag_db_dt_tol = 5.0
                        enable_vert_nhc = true
                        turn_rate_vnhc_gate_deg_s = 3.0
                        turn_rate_compass_gate_deg_s = 3.0
                    },
                    roadIndex,
                    Vector3(initPos[0].toDouble(), initPos[1].toDouble(), initPos[2].toDouble()),
                    Vector3(initVel[0].toDouble(), initVel[1].toDouble(), initVel[2].toDouble()),
                    initHeading,
                    Matrix.identity(3),
                    Vector3(baStat[0].toDouble(), baStat[1].toDouble(), baStat[2].toDouble()),
                    Vector3(0.0, 0.0, 0.0),
                    baselineMag
                )

                val dt = (meta["dt"] as Number).toDouble()

                for (ep in epochs) {
                    if (!isReplayRunning) break

                    @Suppress("UNCHECKED_CAST")
                    val accList = ep["accel"] as List<Number>
                    @Suppress("UNCHECKED_CAST")
                    val gyroList = ep["gyro"] as List<Number>
                    @Suppress("UNCHECKED_CAST")
                    val magList = ep["mag"] as List<Number>
                    val mlSpeed = (ep["ml_speed"] as Number).toDouble()
                    val dbDt = (ep["db_dt"] as Number).toDouble()
                    val psiMagCal = (ep["psi_mag_cal_deg"] as Number).toDouble()
                    val gnssValid = java.lang.Boolean.TRUE == ep["gnss_valid"]

                    var gPosE = 0.0
                    var gPosN = 0.0
                    val gnss = if (gnssValid) {
                        @Suppress("UNCHECKED_CAST")
                        val gPos = ep["gnss_pos_enu"] as List<Number>
                        @Suppress("UNCHECKED_CAST")
                        val gVel = ep["gnss_vel_enu"] as List<Number>
                        gPosE = gPos[0].toDouble()
                        gPosN = gPos[1].toDouble()
                        DeadReckoningEngine.GNSSMeasurement(
                            true,
                            Vector3(gPosE, gPosN, gPos[2].toDouble()),
                            Vector3(gVel[0].toDouble(), gVel[1].toDouble(), gVel[2].toDouble()),
                            3.0
                        )
                    } else {
                        DeadReckoningEngine.GNSSMeasurement(false)
                    }

                    // Update sensor buffers for live diagnostics display
                    curAccel.x = accList[0].toDouble()
                    curAccel.y = accList[1].toDouble()
                    curAccel.z = accList[2].toDouble()
                    curGyro.x = gyroList[0].toDouble()
                    curGyro.y = gyroList[1].toDouble()
                    curGyro.z = gyroList[2].toDouble()
                    curMag.x = magList[0].toDouble()
                    curMag.y = magList[1].toDouble()
                    curMag.z = magList[2].toDouble()

                    val tel = engine!!.step(
                        curAccel, curGyro, mlSpeed, dt, curMag, gnss, psiMagCal, dbDt
                    )

                    mainHandler.post {
                        updateUi(
                            tel, mlSpeed, gnssValid,
                            null, gPosE, gPosN, psiMagCal, dbDt
                        )
                    }

                    Thread.sleep(40L) // 2.5x fast replay
                }

                mainHandler.post {
                    stopReplayDemo()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
        replayThread?.start()
    }

    private fun stopReplayDemo() {
        isReplayRunning = false
        replayThread = null
        btnReplayDemo.text = "▶  START 60s BLACKOUT REPLAY DEMO (MODE A)"
        btnReplayDemo.setBackgroundColor(Color.parseColor("#00E5FF"))
        initEngineFromAssets()
        isHeadingInitialized = false
        lastLoopTimeNs = 0L
    }

    /**
     * Unified UI update consuming the exact same live engine and sensor values.
     * All diagnostics, maps, speed comparisons, and constraint states are transparently displayed.
     */
    private fun updateUi(
        tel: DeadReckoningEngine.NavigationTelemetry,
        mlSpeed: Double,
        gnssValid: Boolean,
        loc: Location?,
        gnssE: Double,
        gnssN: Double,
        psiMagDeg: Double,
        dbDt: Double
    ) {
        // 1. Navigation State Badge & Outage Timer
        if (!gnssValid) {
            badgeNavState.text = "DEAD RECKONING"
            badgeNavState.setTextColor(Color.parseColor("#FF3D00"))
            badgeNavState.setBackgroundResource(R.drawable.badge_outage)

            val outageSec = tel.diagnostics.outage_duration_s
            val mins = (outageSec / 60).toInt()
            val secs = outageSec % 60
            tvOutageTimer.text = String.format(Locale.US, "%02d:%04.1f", mins, secs)
            tvOutageTimer.setTextColor(Color.parseColor("#FF3D00"))
        } else {
            val isReacquiring = (tel.diagnostics.nav_state == "REACQUIRING")
            badgeNavState.text = if (isReacquiring) "REACQUIRING" else "GNSS LOCKED"
            val stateColor = if (isReacquiring) Color.parseColor("#FFAB00") else Color.parseColor("#00E676")
            badgeNavState.setTextColor(stateColor)
            badgeNavState.setBackgroundResource(R.drawable.badge_gnss_locked)
            tvOutageTimer.text = "0.0 s"
            tvOutageTimer.setTextColor(Color.parseColor("#90A4AE"))
        }

        // 2. Local Road Map / ENU Odometry Track View
        val isInsideVta04 = (loc != null && loc.latitude in 52.75..52.90 && loc.longitude in -1.75..-1.55)

        if (isReplayRunning) {
            roadMapView.mapCoverageAvailable = true
            roadMapView.isReferenceMapOnly = false
            roadMapView.mapTitle = "LOCAL OSM ROAD MAP (VTA04 REPLAY)"
        } else if (isInsideVta04) {
            roadMapView.mapCoverageAvailable = true
            roadMapView.isReferenceMapOnly = false
            roadMapView.mapTitle = "LOCAL OSM ROAD MAP (VECTOR ENU)"
        } else {
            roadMapView.mapCoverageAvailable = false
            roadMapView.isReferenceMapOnly = true
            roadMapView.mapTitle = "LOCAL ENU ODOMETRY TRACK"
        }

        roadMapView.estE = tel.pos_enu.x
        roadMapView.estN = tel.pos_enu.y
        roadMapView.headingDeg = tel.heading_deg.toFloat()
        roadMapView.navState = tel.diagnostics.nav_state
        roadMapView.gnssValid = gnssValid
        if (gnssValid) {
            roadMapView.gnssE = gnssE
            roadMapView.gnssN = gnssN
            roadMapView.gnssAccuracyM = loc?.accuracy ?: 3.0f
        }

        if (gnssValid && loc != null) {
            tvGnssLatLon.text = String.format(Locale.US, "LIVE GNSS: %.6f°, %.6f°", loc.latitude, loc.longitude)
            tvGnssAccuracy.text = String.format(Locale.US, "ACC: ±%.1f m", loc.accuracy)
        } else if (gnssValid) {
            tvGnssLatLon.text = String.format(Locale.US, "REPLAY GNSS ENU: E%+.1f m, N%+.1f m", gnssE, gnssN)
            tvGnssAccuracy.text = "ACC: ±3.0 m"
        } else {
            tvGnssLatLon.text = "LIVE GNSS: UNAVAILABLE (DEAD RECKONING)"
            tvGnssAccuracy.text = "ACC: --"
        }

        // 3. Side-by-Side Speed Comparison
        val mlSpeedKmh = mlSpeed * 3.6
        tvSpeedKmh.text = String.format(Locale.US, "%.1f", mlSpeedKmh)
        tvSpeedMs.text = String.format(Locale.US, "%.1f m/s", mlSpeed)

        if (gnssValid && loc != null) {
            val gnssSpdKmh = loc.speed * 3.6
            val gnssSpdMs = loc.speed.toDouble()
            tvGnssSpeedKmh.text = String.format(Locale.US, "%.1f", gnssSpdKmh)
            tvGnssSpeedMs.text = String.format(Locale.US, "%.1f m/s", gnssSpdMs)
            val speedErr = gnssSpdKmh - mlSpeedKmh
            tvSpeedError.text = String.format(Locale.US, "%+.1f km/h", speedErr)
            tvSpeedError.setTextColor(if (Math.abs(speedErr) < 5.0) Color.parseColor("#00E676") else Color.parseColor("#FFAB00"))
        } else {
            tvGnssSpeedKmh.text = "--"
            tvGnssSpeedMs.text = "-- m/s"
            tvSpeedError.text = "--"
            tvSpeedError.setTextColor(Color.parseColor("#90A4AE"))
        }

        // 4. Heading & Compass Display
        compassView.headingDeg = tel.heading_deg.toFloat()
        compassView.headingSigmaDeg = tel.heading_sigma_deg.toFloat()
        compassView.isOutage = !gnssValid

        tvHeadingDeg.text = String.format(Locale.US, "%05.1f°", tel.heading_deg)
        tvHeadingCardinal.text = CompassRoseView.getCardinalDirection(tel.heading_deg.toFloat())

        if (!gnssValid || isOutageSimulated) {
            tvHeadingSource.text = "SOURCE: GYRO DEAD RECKONING"
        } else if (loc != null && loc.speed >= 1.5f && loc.hasBearing()) {
            tvHeadingSource.text = "SOURCE: GNSS-ALIGNED ESKF"
        } else {
            tvHeadingSource.text = "SOURCE: COMPASS-ALIGNED ESKF (STATIONARY)"
        }

        tvFilterHeading.text = String.format(Locale.US, "FILTER HEADING: %05.1f°", tel.heading_deg)
        tvMagHeading.text = String.format(Locale.US, "MAGNETIC: %05.1f°", psiMagDeg)

        // GNSS Course vs Filter Heading Comparison
        if (gnssValid && loc != null && loc.speed >= 1.5f && loc.hasBearing()) {
            val gnssCourse = (loc.bearing.toDouble() % 360.0 + 360.0) % 360.0
            tvGnssCourse.text = String.format(Locale.US, "GNSS COURSE: %05.1f°", gnssCourse)
            var diff = (tel.heading_deg - gnssCourse) % 360.0
            if (diff > 180.0) diff -= 360.0 else if (diff < -180.0) diff += 360.0
            tvHeadingDelta.text = String.format(Locale.US, "Δ vs GNSS: %+5.1f°", diff)
        } else if (gnssValid && loc != null && loc.speed < 1.5f) {
            tvGnssCourse.text = "GNSS COURSE: STATIONARY / NO COURSE"
            var magDiff = (tel.heading_deg - psiMagDeg) % 360.0
            if (magDiff > 180.0) magDiff -= 360.0 else if (magDiff < -180.0) magDiff += 360.0
            tvHeadingDelta.text = String.format(Locale.US, "Δ vs MAG: %+5.1f°", magDiff)
        } else {
            tvGnssCourse.text = "GNSS COURSE: --"
            tvHeadingDelta.text = "Δ: --"
        }

        // 5. Phone Tilt / Attitude & Altitude / Vertical Motion
        if (isMagValid) {
            horizonView.pitchDeg = livePitchDeg
            horizonView.rollDeg = liveRollDeg
            horizonView.phoneYawDeg = psiMagDeg.toFloat()
            tvPhoneAttitude.text = String.format(Locale.US, "R: %+5.1f° | P: %+5.1f°", liveRollDeg, livePitchDeg)
        } else {
            val phoneRollDeg = Math.toDegrees(Math.atan2(curAccel.y, curAccel.z)).toFloat()
            val phonePitchDeg = Math.toDegrees(Math.atan2(-curAccel.x, Math.sqrt(curAccel.y * curAccel.y + curAccel.z * curAccel.z))).toFloat()
            horizonView.rollDeg = phoneRollDeg
            horizonView.pitchDeg = phonePitchDeg
            horizonView.phoneYawDeg = psiMagDeg.toFloat()
            tvPhoneAttitude.text = String.format(Locale.US, "R: %+5.1f° | P: %+5.1f°", phoneRollDeg, phonePitchDeg)
        }

        if (gnssValid && loc != null) {
            tvGnssAlt.text = String.format(Locale.US, "GNSS ALT: %.1f m", loc.altitude)
        } else {
            tvGnssAlt.text = "GNSS ALT: --"
        }
        tvFilterUp.text = String.format(Locale.US, "LOCAL ΔUP: %+5.1f m", tel.pos_enu.z)
        tvVertSpeed.text = String.format(Locale.US, "VERT SPEED: %+5.2f m/s", tel.vel_enu.z)

        // Diagnostic Vertical Drift Alert (> 15m)
        if (Math.abs(tel.pos_enu.z) > 15.0) {
            tvVerticalDriftWarn.visibility = View.VISIBLE
        } else {
            tvVerticalDriftWarn.visibility = View.GONE
        }

        // 6. Sensor Health Chips & Measured Rate
        val aNorm = curAccel.norm()
        val gNormDeg = Math.toDegrees(curGyro.norm())
        val mNorm = curMag.norm()

        tvNormAccel.text = String.format(Locale.US, "Acc: %.2f m/s²", aNorm)
        tvNormGyro.text = String.format(Locale.US, "Gyro: %.2f °/s", gNormDeg)
        tvNormMag.text = String.format(Locale.US, "Mag: %.1f µT", mNorm)

        tvSensorRate.text = String.format(Locale.US, "RATE: %.2f Hz | Δt: %.0f ms", measuredHz, measuredDtMs)

        updateSensorChip(tileSensorAccel, "ACCEL", Math.abs(aNorm - 9.81) < 15.0)
        updateSensorChip(tileSensorGyro, "GYRO", gNormDeg < 500.0)
        updateSensorChip(tileSensorMag, "MAG", mNorm in 15.0..80.0)
        updateSensorChip(tileSensorGnss, "GNSS", gnssValid)

        // 7. Inertial Constraints & Quality Gates
        if (tel.diagnostics.nhc_active) {
            tileNhc.text = "LATERAL NHC: ACTIVE"
            tileNhc.setTextColor(Color.parseColor("#00E676"))
        } else {
            tileNhc.text = "LATERAL NHC: INACTIVE"
            tileNhc.setTextColor(Color.parseColor("#546E7A"))
        }

        if (tel.diagnostics.vnhc_active) {
            tileVnhc.text = "VERTICAL NHC: ACTIVE"
            tileVnhc.setTextColor(Color.parseColor("#00E676"))
        } else {
            tileVnhc.text = "VERTICAL NHC: INACTIVE"
            tileVnhc.setTextColor(Color.parseColor("#546E7A"))
        }

        if (tel.diagnostics.compass_active) {
            tileCompass.text = "COMPASS: ACCEPTED"
            tileCompass.setTextColor(Color.parseColor("#00E676"))
        } else {
            val turnRate = Math.toDegrees(Math.abs(curGyro.z))
            val reason = if (turnRate > 3.0) "TURN RATE" else if (dbDt > 5.0) "DISTURBANCE" else "GATE"
            tileCompass.text = "COMPASS: REJ ($reason)"
            tileCompass.setTextColor(Color.parseColor("#FFAB00"))
        }

        if (isReplayRunning || isInsideVta04) {
            if (tel.diagnostics.map_active) {
                tileOsm.text = "OSM MAP: ACCEPTED"
                tileOsm.setTextColor(Color.parseColor("#00E676"))
            } else {
                tileOsm.text = "OSM MAP: SEARCHING"
                tileOsm.setTextColor(Color.parseColor("#546E7A"))
            }
        } else {
            tileOsm.text = "OSM MAP: OUT OF REGION"
            tileOsm.setTextColor(Color.parseColor("#546E7A"))
        }

        tvBiasAccel.text = String.format(Locale.US, "Accel Bias b_ax: %+.3f m/s²", tel.diagnostics.b_accel_x_applied)
        tvBiasGyro.text = String.format(Locale.US, "ZARU Gyro b_gz: %+.3f°/s", Math.toDegrees(tel.diagnostics.b_gyro_z_applied))

        // 8. Local Cartesian ENU Position
        tvPosSigma.text = String.format(Locale.US, "±%.2f m", tel.pos_sigma_m)
        tvPosE.text = String.format(Locale.US, "%+.1f m", tel.pos_enu.x)
        tvPosN.text = String.format(Locale.US, "%+.1f m", tel.pos_enu.y)
        tvPosU.text = String.format(Locale.US, "%+.1f m", tel.pos_enu.z)
    }

    private fun updateSensorChip(view: TextView, label: String, ok: Boolean) {
        if (ok) {
            view.text = "$label | OK"
            view.setTextColor(Color.parseColor("#00E676"))
            view.setBackgroundResource(R.drawable.badge_gnss_locked)
        } else {
            view.text = "$label | LOST"
            view.setTextColor(Color.parseColor("#FF5252"))
            view.setBackgroundResource(R.drawable.badge_outage)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        liveLoopHandler.removeCallbacks(liveLoopRunnable)
        sensorLogger.stopLogging()
        rawCalibrationLogger.stopLogging()
        stopReplayDemo()
        sensorManager.unregisterListener(this)
        locationManager?.removeUpdates(this)
    }
}
