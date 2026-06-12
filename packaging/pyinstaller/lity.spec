# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path(SPECPATH).parents[1]

# Package data. The web UI build (web_dist) is bundled when present so the
# pywebview interface works from a file:// URL; the Qt build still works
# without it. Build it first with `cd frontend && npm install && npm run build`.
WEB_DIST = ROOT / "src" / "lity" / "interfaces" / "desktop_web" / "web_dist"
datas = [(str(ROOT / "src" / "lity" / "resources"), "lity/resources")]
if WEB_DIST.exists():
    datas.append((str(WEB_DIST), "lity/interfaces/desktop_web/web_dist"))

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT), str(ROOT / "src")],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "ctranslate2",
        "fastembed",
        "faster_whisper",
        "numpy",
        "onnx",
        "onnxruntime",
        "PIL",
        "piper",
        "sounddevice",
        "tokenizers",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Lity",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Lity",
)
