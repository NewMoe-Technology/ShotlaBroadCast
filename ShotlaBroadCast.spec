# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ShotlaBroadCast.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['soundfile', 'numpy', 'onnxruntime', 'pyworld', 'resampy', 'loguru', 'wmi', 'pywin32', 'pyopencl'],
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
