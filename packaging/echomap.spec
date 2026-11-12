# PyInstaller spec for EchoMap (one-folder build, windowed). Run from the repository root:
#   pyinstaller --noconfirm --clean packaging/echomap.spec
# Kept small on purpose: models are NOT bundled (separate download), and scipy, shapely and PyMuPDF
# (build-time/dev only) are excluded.
import os
from PyInstaller.utils.hooks import collect_submodules

root = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [(os.path.join(root, "echomap", "web"), "echomap/web"),
         (os.path.join(root, "echomap", "data"), "echomap/data")]
hidden = collect_submodules("webview.platforms") + ["clr_loader", "pythonnet"] + collect_submodules("echomap")

a = Analysis([os.path.join(root, "packaging", "launcher.py")], pathex=[root], datas=datas, hiddenimports=hidden,
             excludes=["scipy", "shapely", "pymupdf", "fitz", "tkinter", "matplotlib", "pytest", "IPython",
                       "pandas", "PIL.ImageQt", "PyQt5", "PyQt6", "PySide6", "cefpython3"],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="EchoMap", console=False,
          icon=os.path.join(root, "packaging", "echomap.ico"), version=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="EchoMap")
