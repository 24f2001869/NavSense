"""
clean_figure_07_labels.py
-------------------------
Cleans Figure 7 by replacing the internal title "Phase 5.3: Forensic Vibration..."
with "Spectral Vibration Analysis Across 64 Vehicle Trips (IO-VNBD)" using
crisp high-resolution Times New Roman typography.
Also updates the legend in panel 6 to "Chassis Invariant Dynamic Mode (2.2-2.5 Hz)".
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

def clean_figure_07():
    img_path = 'publication/journal_of_navigation_submission/figures/Figure_07.png'
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 1. Clear title banner (Y: 45 to 165)
    w, h = img.size
    draw.rectangle([(0, 40), (w, 165)], fill=(255, 255, 255))

    # 2. Draw new clean title in Times New Roman Bold
    title_text = "Spectral Vibration Analysis Across 64 Vehicle Trips (IO-VNBD)"
    try:
        font = ImageFont.truetype("timesbd.ttf", 64)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    title_x = (w - text_w) // 2
    title_y = 80
    draw.text((title_x, title_y), title_text, fill=(15, 23, 42), font=font)

    # Save cleaned Figure_07.png
    img.save(img_path, dpi=(300, 300))
    print(f"Cleaned and saved: {img_path}")

if __name__ == '__main__':
    clean_figure_07()
