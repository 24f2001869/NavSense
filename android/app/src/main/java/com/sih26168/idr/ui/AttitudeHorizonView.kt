package com.sih26168.idr.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import java.util.Locale

/**
 * Visual artificial-horizon style attitude indicator showing real phone spatial tilt.
 * Visibly rotates with phone roll and shifts with phone pitch.
 * Labeled explicitly as "PHONE ATTITUDE" to avoid confounding with vehicle navigation heading.
 */
class AttitudeHorizonView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var rollDeg: Float = 0.0f
        set(value) { field = value; postInvalidateOnAnimation() }

    var pitchDeg: Float = 0.0f
        set(value) { field = value; postInvalidateOnAnimation() }

    var phoneYawDeg: Float = 0.0f
        set(value) { field = value; postInvalidateOnAnimation() }

    private val skyPaint = Paint().apply {
        color = Color.parseColor("#152238")
        style = Paint.Style.FILL
    }
    private val groundPaint = Paint().apply {
        color = Color.parseColor("#1E1B18")
        style = Paint.Style.FILL
    }
    private val horizonLinePaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        strokeWidth = 3.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val pitchLadderPaint = Paint().apply {
        color = Color.parseColor("#80FFFFFF")
        strokeWidth = 2.0f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val reticlePaint = Paint().apply {
        color = Color.parseColor("#FFAB00")
        strokeWidth = 3.5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.parseColor("#E0E0E0")
        textSize = 24f
        typeface = Typeface.create(Typeface.MONOSPACE, Typeface.BOLD)
        isAntiAlias = true
    }
    private val labelPaint = Paint().apply {
        color = Color.parseColor("#00E5FF")
        textSize = 20f
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        letterSpacing = 0.08f
        isAntiAlias = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val w = width.toFloat()
        val h = height.toFloat()
        val cx = w / 2f
        val cy = h / 2f

        // Clip to rounded boundary
        val borderPath = Path().apply {
            addRoundRect(0f, 0f, w, h, 16f, 16f, Path.Direction.CW)
        }
        canvas.clipPath(borderPath)

        // Pitch scale: 2.0 pixels per degree of pitch
        val pitchShiftPx = pitchDeg * 2.0f

        canvas.save()
        // Rotate opposite to roll to simulate true artificial horizon
        canvas.rotate(-rollDeg, cx, cy)
        canvas.translate(0f, pitchShiftPx)

        // 1. Sky & Ground Rectangles
        val bigR = Math.max(w, h) * 2.0f
        canvas.drawRect(cx - bigR, cy - bigR, cx + bigR, cy, skyPaint)
        canvas.drawRect(cx - bigR, cy, cx + bigR, cy + bigR, groundPaint)

        // 2. Horizon Line
        canvas.drawLine(cx - bigR, cy, cx + bigR, cy, horizonLinePaint)

        // 3. Pitch Ladder Lines (±10°, ±20°)
        val pitchLevels = intArrayOf(-20, -10, 10, 20)
        for (p in pitchLevels) {
            val yLevel = cy - (p * 2.0f)
            val lineHalfW = if (p % 20 == 0) 45f else 25f
            canvas.drawLine(cx - lineHalfW, yLevel, cx + lineHalfW, yLevel, pitchLadderPaint)
            val pText = "${Math.abs(p)}°"
            canvas.drawText(pText, cx + lineHalfW + 6f, yLevel + 8f, pitchLadderPaint)
        }

        canvas.restore()

        // 4. Fixed Aircraft / Phone Reticle Center Symbol
        canvas.drawLine(cx - 36f, cy, cx - 12f, cy, reticlePaint)
        canvas.drawLine(cx + 12f, cy, cx + 36f, cy, reticlePaint)
        canvas.drawLine(cx, cy - 8f, cx, cy + 8f, reticlePaint)

        // 5. Title & Numerical Readout Overlay
        canvas.drawText("PHONE ATTITUDE", 16f, 28f, labelPaint)
        val attitudeStr = String.format(Locale.US, "R:%+5.1f° P:%+5.1f°", rollDeg, pitchDeg)
        canvas.drawText(attitudeStr, 16f, h - 14f, textPaint)
    }
}
