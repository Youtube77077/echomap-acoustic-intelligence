"""Generate packaging/echomap.ico (run in CI before PyInstaller)."""
from pathlib import Path
from PIL import Image, ImageDraw

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.ellipse([8, 8, S - 8, S - 8], fill=(13, 21, 36, 255), outline=(79, 140, 255, 255), width=10)
for r, a in ((92, 120), (64, 180), (36, 255)):
    d.arc([S / 2 - r, S / 2 - r, S / 2 + r, S / 2 + r], 200, 340, fill=(255, 179, 64, a), width=12)
d.ellipse([S / 2 - 18, S / 2 - 18, S / 2 + 18, S / 2 + 18], fill=(255, 90, 60, 255))
img.save(Path(__file__).with_name("echomap.ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
