package com.sih26168.idr.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.sih26168.idr.engine.RoadNetworkIndex
import java.util.*
import kotlin.math.*

/**
 * High-performance 2D Vector Canvas rendering local OSM road segments,
 * live GNSS position (Layer A), and estimated ESKF vehicle trajectory (Layer B).
 * Explicitly labeled as "LOCAL OSM ROAD MAP (VECTOR ENU)".
 */
class LocalRoadMapView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // Road segments from local offline OSM data
    private var segments: List<RoadNetworkIndex.Segment>? = null

    // Geographic Map Coverage Status
    var mapCoverageAvailable: Boolean = false
        set(value) { field = value; postInvalidateOnAnimation() }
    var isReferenceMapOnly: Boolean = false
        set(value) { field = value; postInvalidateOnAnimation() }
    var mapTitle: String = "LOCAL ENU ODOMETRY TRACK"
        set(value) { field = value; postInvalidateOnAnimation() }

    // Live GNSS State (Layer A)
    var gnssValid: Boolean = false
        set(value) { field = value; postInvalidateOnAnimation() }
    var gnssE: Double = 0.0
        set(value) { field = value; postInvalidateOnAnimation() }
    var gnssN: Double = 0.0
        set(value) { field = value; postInvalidateOnAnimation() }
    var gnssAccuracyM: Float = 5.0f
        set(value) { field = value; postInvalidateOnAnimation() }

    // Estimated Filter Navigation State (Layer B)
    var estE: Double = 0.0
        set(value) {
            field = value
            addTrailPoint(value, estN)
            postInvalidateOnAnimation()
        }
    var estN: Double = 0.0
        set(value) {
            field = value
            addTrailPoint(estE, value)
            postInvalidateOnAnimation()
        }
    var headingDeg: Float = 0.0f
        set(value) { field = value; postInvalidateOnAnimation() }

    var navState: String = "GNSS_LOCKED"
        set(value) { field = value; postInvalidateOnAnimation() }

    // View scale: pixels per meter (default: 1.0 px/m -> 200m visible radius)
    private var scalePxPerMeter: Float = 1.2f

    // Estimated trajectory trail buffer (last 100 points)
    private val trailPoints = LinkedList<PointF>()
    private val maxTrailPoints = 100

    // Paints
    private val bgPaint = Paint().apply {
        color = Color.parseColor("#0A0E17")
        style = Paint.Style.FILL
    }
    private val gridPaint = Paint().apply {
        color = Color.parseColor("#152238")
        strokeWidth = 1.0f
        style = Paint.Style.STROKE
    }
    private val roadPaint = Paint().apply {
        color = Color.parseColor("#1E2E48")
        strokeWidth = 5.0f
        strokeCap = Paint.Cap.ROUND
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val roadCenterPaint = Paint().apply {
        color = Color.parseColor("#2E4466")
        strokeWidth = 2.0f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val gnssPointPaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val gnssRingPaint = Paint().apply {
        color = Color.parseColor("#3300E5FF")
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val gnssRingStrokePaint = Paint().apply {
        color = Color.parseColor("#8000E5FF")
        strokeWidth = 1.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val estVehiclePaint = Paint().apply {
        color = Color.parseColor("#FFAB00")
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val estVehicleStrokePaint = Paint().apply {
        color = Color.parseColor("#FFFFFF")
        strokeWidth = 2.0f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val trailPaint = Paint().apply {
        color = Color.parseColor("#80FFAB00")
        strokeWidth = 3.0f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.parseColor("#90A4AE")
        textSize = 28f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
        isAntiAlias = true
    }
    private val titlePaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        textSize = 26f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        letterSpacing = 0.08f
        isAntiAlias = true
    }
    private val northPaint = Paint().apply {
        color = Color.parseColor("#FF5252")
        textSize = 30f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }
    private val originPaint = Paint().apply {
        color = Color.parseColor("#8000E5FF")
        strokeWidth = 2.0f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val originTextPaint = Paint().apply {
        color = Color.parseColor("#8000E5FF")
        textSize = 18f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.NORMAL)
        isAntiAlias = true
    }
    private val warnPaint = Paint().apply {
        color = Color.parseColor("#FFAB00")
        textSize = 18f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        isAntiAlias = true
    }

    fun setRoadNetwork(segments: List<RoadNetworkIndex.Segment>?) {
        this.segments = segments
        postInvalidateOnAnimation()
    }

    private fun addTrailPoint(e: Double, n: Double) {
        trailPoints.addLast(PointF(e.toFloat(), n.toFloat()))
        if (trailPoints.size > maxTrailPoints) {
            trailPoints.removeFirst()
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2f
        val cy = h / 2f

        // 1. Background
        canvas.drawRect(0f, 0f, w, h, bgPaint)

        // Center on estimated position
        val centerE = estE
        val centerN = estN

        // 2. Coordinate Grid (50 meter increments)
        val gridStepM = 50.0
        val gridStepPx = (gridStepM * scalePxPerMeter).toFloat()
        val startX = (cx - (centerE % gridStepM) * scalePxPerMeter).toFloat()
        val startY = (cy + (centerN % gridStepM) * scalePxPerMeter).toFloat()

        var x = startX - (cx / gridStepPx).toInt() * gridStepPx - gridStepPx
        while (x < w + gridStepPx) {
            canvas.drawLine(x, 0f, x, h, gridPaint)
            x += gridStepPx
        }
        var y = startY - (cy / gridStepPx).toInt() * gridStepPx - gridStepPx
        while (y < h + gridStepPx) {
            canvas.drawLine(0f, y, w, y, gridPaint)
            y += gridStepPx
        }

        // 3. Render Offline Road Network Segments ONLY IF geographic coverage is valid for this region
        if (mapCoverageAvailable && segments != null) {
            segments?.let { segList ->
                for (seg in segList) {
                    val x1 = cx + ((seg.p1_e - centerE) * scalePxPerMeter).toFloat()
                    val y1 = cy - ((seg.p1_n - centerN) * scalePxPerMeter).toFloat()
                    val x2 = cx + ((seg.p2_e - centerE) * scalePxPerMeter).toFloat()
                    val y2 = cy - ((seg.p2_n - centerN) * scalePxPerMeter).toFloat()

                    // Frustum culling check
                    if ((x1 in -50f..(w + 50f) && y1 in -50f..(h + 50f)) ||
                        (x2 in -50f..(w + 50f) && y2 in -50f..(h + 50f))) {
                        canvas.drawLine(x1, y1, x2, y2, roadPaint)
                        canvas.drawLine(x1, y1, x2, y2, roadCenterPaint)
                    }
                }
            }
        } else {
            // Draw origin crosshair at (0, 0) relative to start fix
            val ox = cx + ((0.0 - centerE) * scalePxPerMeter).toFloat()
            val oy = cy - ((0.0 - centerN) * scalePxPerMeter).toFloat()
            if (ox in -40f..(w + 40f) && oy in -40f..(h + 40f)) {
                canvas.drawCircle(ox, oy, 6f, originPaint)
                canvas.drawLine(ox - 16f, oy, ox + 16f, oy, originPaint)
                canvas.drawLine(ox, oy - 16f, ox, oy + 16f, originPaint)
                canvas.drawText("ORIGIN (0,0)", ox + 10f, oy - 10f, originTextPaint)
            }
        }

        // 4. Render Estimated Trajectory Trail
        if (trailPoints.size > 1) {
            val path = Path()
            var first = true
            for (pt in trailPoints) {
                val px = cx + ((pt.x - centerE) * scalePxPerMeter).toFloat()
                val py = cy - ((pt.y - centerN) * scalePxPerMeter).toFloat()
                if (first) {
                    path.moveTo(px, py)
                    first = false
                } else {
                    path.lineTo(px, py)
                }
            }
            canvas.drawPath(path, trailPaint)
        }

        // 5. Layer A: Live GNSS Position Marker
        if (gnssValid) {
            val gx = cx + ((gnssE - centerE) * scalePxPerMeter).toFloat()
            val gy = cy - ((gnssN - centerN) * scalePxPerMeter).toFloat()
            val accPx = (gnssAccuracyM * scalePxPerMeter).coerceIn(12f, 160f)

            // Accuracy disk and ring
            canvas.drawCircle(gx, gy, accPx, gnssRingPaint)
            canvas.drawCircle(gx, gy, accPx, gnssRingStrokePaint)

            // GNSS center dot
            canvas.drawCircle(gx, gy, 10f, gnssPointPaint)

            // GNSS label
            canvas.drawText("GNSS FIX", gx + 14f, gy - 12f, textPaint)
        } else {
            // Label outage state on map
            val statusPaint = Paint(textPaint).apply {
                color = Color.parseColor("#FF5252")
                textSize = 24f
                typeface = Typeface.DEFAULT_BOLD
            }
            canvas.drawText("GNSS: UNAVAILABLE (DEAD RECKONING ACTIVE)", 24f, h - 50f, statusPaint)
        }

        // 6. Layer B: Estimated Vehicle Marker (Always at cx, cy by auto-center)
        canvas.save()
        canvas.translate(cx, cy)
        canvas.rotate(headingDeg)

        // Draw vehicle triangle pointer
        val vPath = Path().apply {
            moveTo(0f, -22f) // Forward tip
            lineTo(-14f, 16f) // Left rear
            lineTo(0f, 10f)   // Notch
            lineTo(14f, 16f)  // Right rear
            close()
        }
        val vColor = if (navState == "DEAD_RECKONING") Color.parseColor("#FF5252") else Color.parseColor("#FFAB00")
        estVehiclePaint.color = vColor
        canvas.drawPath(vPath, estVehiclePaint)
        canvas.drawPath(vPath, estVehicleStrokePaint)
        canvas.restore()

        // 7. Map Title & Legend Overlays
        canvas.drawText(mapTitle, 24f, 38f, titlePaint)

        if (!mapCoverageAvailable) {
            canvas.drawText("⚠ LIVE MAP UNAVAILABLE FOR CURRENT REGION", 24f, 62f, warnPaint)
            canvas.drawText("NO LOCAL OFFLINE OSM TILES • SHOWING ENU ODOMETRY", 24f, 82f, textPaint)
        } else {
            val subPaint = Paint(textPaint).apply {
                color = Color.parseColor("#00E676")
                textSize = 18f
            }
            canvas.drawText("OFFLINE VECTOR ROAD NETWORK", 24f, 62f, subPaint)
        }

        // North Indicator (Top Right)
        val nx = w - 60f
        val ny = 45f
        canvas.drawText("▲ N", nx, ny, northPaint)

        // Scale Indicator (Bottom Left)
        val scaleLenM = 50f
        val scalePx = scaleLenM * scalePxPerMeter
        val bx = 24f
        val by = h - 20f
        canvas.drawLine(bx, by, bx + scalePx, by, textPaint)
        canvas.drawLine(bx, by - 6f, bx, by + 6f, textPaint)
        canvas.drawLine(bx + scalePx, by - 6f, bx + scalePx, by + 6f, textPaint)
        canvas.drawText("${scaleLenM.toInt()} m", bx + scalePx + 12f, by + 8f, textPaint)

        // Zoom Scale Text (Bottom Right)
        val zoomText = String.format(Locale.US, "E:%+.0f N:%+.0f", centerE, centerN)
        canvas.drawText(zoomText, w - 240f, by + 8f, textPaint)
    }
}
