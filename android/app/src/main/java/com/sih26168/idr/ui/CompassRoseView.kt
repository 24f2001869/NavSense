package com.sih26168.idr.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/**
 * Custom High-Aesthetic Automotive HUD Compass Rose.
 * Renders rotating azimuth, cardinal marks, and 1-sigma uncertainty wedge.
 */
class CompassRoseView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var headingDeg: Float = 0f
        set(value) {
            field = (value % 360f + 360f) % 360f
            invalidate()
        }

    var headingSigmaDeg: Float = 5f
        set(value) {
            field = value.coerceIn(0.1f, 45f)
            invalidate()
        }

    var isOutage: Boolean = false
        set(value) {
            field = value
            invalidate()
        }

    private val ringPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.parseColor("#1F2E45")
    }

    private val tickPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f
        color = Color.parseColor("#90A4AE")
    }

    private val northPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#FF3D00")
        textSize = 28f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }

    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#90A4AE")
        textSize = 24f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }

    private val intercardinalPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#546E7A")
        textSize = 18f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT
    }

    companion object {
        fun getCardinalDirection(deg: Float): String {
            val d = (deg % 360f + 360f) % 360f
            return when {
                d >= 337.5f || d < 22.5f -> "NORTH"
                d < 67.5f -> "NORTHEAST"
                d < 112.5f -> "EAST"
                d < 157.5f -> "SOUTHEAST"
                d < 202.5f -> "SOUTH"
                d < 247.5f -> "SOUTHWEST"
                d < 292.5f -> "WEST"
                else -> "NORTHWEST"
            }
        }
    }

    private val needlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL_AND_STROKE
        strokeWidth = 4f
        color = Color.parseColor("#00E5FF")
    }

    private val uncertaintyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#2600E5FF") // 15% opacity cyan
    }

    private val centerTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textSize = 36f
        textAlign = Paint.Align.CENTER
        typeface = Typeface.DEFAULT_BOLD
    }

    private val sigmaTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#00E5FF")
        textSize = 20f
        textAlign = Paint.Align.CENTER
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val cx = width / 2f
        val cy = height / 2f
        val radius = (Math.min(width, height) / 2f) - 16f
        if (radius <= 0f) return

        // 1. Draw outer ring
        canvas.drawCircle(cx, cy, radius, ringPaint)
        canvas.drawCircle(cx, cy, radius * 0.75f, ringPaint)

        // 2. Draw 1-sigma uncertainty wedge centered at 12 o'clock (top)
        val arcRect = RectF(cx - radius, cy - radius, cx + radius, cy + radius)
        val sweepAngle = headingSigmaDeg * 2f
        val startAngle = 270f - headingSigmaDeg
        uncertaintyPaint.color = if (isOutage) Color.parseColor("#33FFAB00") else Color.parseColor("#2600E5FF")
        canvas.drawArc(arcRect, startAngle, sweepAngle, true, uncertaintyPaint)

        // 3. Draw rotating cardinal dial
        canvas.save()
        // Rotate canvas opposite to vehicle heading so 12 o'clock represents vehicle forward
        canvas.rotate(-headingDeg, cx, cy)

        val cardinals = mapOf(
            0 to "N", 45 to "NE", 90 to "E", 135 to "SE",
            180 to "S", 225 to "SW", 270 to "W", 315 to "NW"
        )
        for (i in 0 until 360 step 15) {
            val rad = Math.toRadians(i.toDouble() - 90.0)
            val cosR = cos(rad).toFloat()
            val sinR = sin(rad).toFloat()

            val isCardinal = (i % 90 == 0)
            val isIntercardinal = (i % 45 == 0 && !isCardinal)
            val innerR = if (isCardinal) radius - 24f else if (isIntercardinal) radius - 16f else radius - 8f
            canvas.drawLine(
                cx + innerR * cosR, cy + innerR * sinR,
                cx + radius * cosR, cy + radius * sinR,
                tickPaint
            )

            if (cardinals.containsKey(i)) {
                val labelR = radius - 38f
                val labelX = cx + labelR * cosR
                val labelY = cy + labelR * sinR + 8f
                val text = cardinals[i]!!
                val p = when (text) {
                    "N" -> northPaint
                    "E", "S", "W" -> labelPaint
                    else -> intercardinalPaint
                }
                canvas.drawText(text, labelX, labelY, p)

                // Prominent red directional triangle for True North (0°) on the rotating dial
                if (i == 0) {
                    val nArrowPath = Path().apply {
                        val tipR = radius - 2f
                        val baseR = radius - 16f
                        val perpCos = -sinR
                        val perpSin = cosR
                        moveTo(cx + tipR * cosR, cy + tipR * sinR)
                        lineTo(cx + baseR * cosR + 7f * perpCos, cy + baseR * sinR + 7f * perpSin)
                        lineTo(cx + baseR * cosR - 7f * perpCos, cy + baseR * sinR - 7f * perpSin)
                        close()
                    }
                    canvas.drawPath(nArrowPath, northPaint)
                }
            }
        }
        canvas.restore()

        // 4. Draw fixed forward needle at 12 o'clock
        val needlePath = Path().apply {
            moveTo(cx, cy - radius + 5f)
            lineTo(cx - 10f, cy - radius + 28f)
            lineTo(cx + 10f, cy - radius + 28f)
            close()
        }
        needlePaint.color = if (isOutage) Color.parseColor("#FFAB00") else Color.parseColor("#00E5FF")
        canvas.drawPath(needlePath, needlePaint)

        // 5. Draw center heading readout
        val hInt = headingDeg.toInt()
        canvas.drawText("${hInt}°", cx, cy + 8f, centerTextPaint)
        canvas.drawText("±${String.format("%.1f", headingSigmaDeg)}°", cx, cy + 32f, sigmaTextPaint)
    }
}
