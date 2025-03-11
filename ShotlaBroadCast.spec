# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ShotlaBroadCast.py'],
    pathex=[],
    binaries=[('C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin\\cudnn_ops_infer64_8.dll', '.'), ('D:\\zlib123dllx64\\dll_x64\\zlibwapi.dll', '.')],
    datas=[],
    hiddenimports=['modelscope', 'wmi', 'pywin32', 'fastapi', 'fastapi.middleware.cors', 'fastapi.responses', 'scipy', 'loguru', 'pynvml', 'pyworld', 'librosa', 'numpy', 'onnxruntime', 'uvicorn'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ShotlaBroadCast',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'],
)
