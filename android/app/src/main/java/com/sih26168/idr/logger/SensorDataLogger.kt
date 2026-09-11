package com.sih26168.idr.logger

import android.content.Context
import android.util.Log
import com.sih26168.idr.engine.DeadReckoningEngine
import com.sih26168.idr.engine.Vector3
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * High-frequency telemetry and sensor logger for real Android phone field validation (Stage C10.2).
 * Logs raw IMU, Magnetometer, GNSS, ML speed, and ESKF diagnostics at 10 Hz into CSV format.
 */
class SensorDataLogger(private val context: Context) {

    companion object {
        private const val TAG = "SensorDataLogger"
        private const val CSV_HEADER = "timestamp_ms,dt_s," +
                "accel_x,accel_y,accel_z," +
                "gyro_x,gyro_y,gyro_z," +
                "mag_x,mag_y,mag_z," +
                "grav_x,grav_y,grav_z," +
                "gnss_lat,gnss_lon,gnss_alt,gnss_speed,gnss_bearing,gnss_accuracy," +
                "engine_state,ml_speed,heading_deg,pos_e,pos_n,pos_u," +
                "nhc_enabled,vnhc_enabled,compass_accepted,map_accepted"
    }

    private var writer: PrintWriter? = null
    private var currentFile: File? = null
    var isLogging: Boolean = false
        private set

    fun startLogging(): File? {
        if (isLogging) return currentFile

        return try {
            val dir = context.getExternalFilesDir(null) ?: context.filesDir
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val file = File(dir, "idr_telemetry_$timeStamp.csv")
            writer = PrintWriter(BufferedWriter(FileWriter(file)))
            writer?.println(CSV_HEADER)
            writer?.flush()
            currentFile = file
            isLogging = true
            Log.i(TAG, "Started telemetry logging to: ${file.absolutePath}")
            file
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start logging", e)
            null
        }
    }

    fun logEpoch(
        timestampMs: Long,
        dtSec: Double,
        accel: Vector3,
        gyro: Vector3,
        mag: Vector3,
        gravity: Vector3,
        gnssLat: Double,
        gnssLon: Double,
        gnssAlt: Double,
        gnssSpeed: Double,
        gnssBearing: Double,
        gnssAccuracy: Double,
        tel: DeadReckoningEngine.NavigationTelemetry,
        effectiveSpeedEst: Double
    ) {
        if (!isLogging || writer == null) return

        try {
            val line = String.format(
                Locale.US,
                "%d,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.2f,%.2f,%.2f,%.4f,%.4f,%.4f,%.7f,%.7f,%.2f,%.2f,%.1f,%.2f,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%b,%b,%b,%b",
                timestampMs,
                dtSec,
                accel.x, accel.y, accel.z,
                gyro.x, gyro.y, gyro.z,
                mag.x, mag.y, mag.z,
                gravity.x, gravity.y, gravity.z,
                gnssLat, gnssLon, gnssAlt, gnssSpeed, gnssBearing, gnssAccuracy,
                tel.diagnostics.nav_state,
                effectiveSpeedEst,
                tel.heading_deg,
                tel.pos_enu.x, tel.pos_enu.y, tel.pos_enu.z,
                tel.diagnostics.nhc_active,
                tel.diagnostics.vnhc_active,
                tel.diagnostics.compass_active,
                tel.diagnostics.map_active
            )
            writer?.println(line)
        } catch (e: Exception) {
            Log.e(TAG, "Error writing telemetry row", e)
        }
    }

    fun stopLogging(): File? {
        if (!isLogging) return null

        val file = currentFile
        try {
            writer?.flush()
            writer?.close()
            writer = null
            isLogging = false
            Log.i(TAG, "Stopped telemetry logging. File closed: ${file?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing telemetry file", e)
        }
        currentFile = null
        return file
    }
}
