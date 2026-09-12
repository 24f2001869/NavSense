"""
convert_figures_to_tiff.py
--------------------------
Converts PNG figures in publication/journal_of_navigation_submission/figures/
to production-quality TIFF format (LZW compression, high DPI) in accordance with
Cambridge University Press Journal of Navigation guidelines.
"""

import os
from PIL import Image

def convert_figures():
    fig_dir = "publication/journal_of_navigation_submission/figures"
    for i in range(1, 9):
        png_path = os.path.join(fig_dir, f"Figure_{i:02d}.png")
        tif_path = os.path.join(fig_dir, f"Figure_{i:02d}.tif")
        
        if not os.path.exists(png_path):
            print(f"Warning: {png_path} not found.")
            continue
            
        with Image.open(png_path) as img:
            # Convert RGBA to RGB if saving TIFF without alpha channel for print compliance
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                bg = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                bg.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                save_img = bg
            else:
                save_img = img.convert('RGB')
                
            # Cambridge recommends 300 DPI for halftone, 800-1200 DPI for line/combination
            # Save at 600 DPI with LZW lossless compression
            dpi = (600, 600)
            save_img.save(
                tif_path,
                format='TIFF',
                dpi=dpi,
                compression='tiff_lzw'
            )
            print(f"Created: {tif_path} | Size: {save_img.size} | DPI: {dpi}")

if __name__ == '__main__':
    convert_figures()
