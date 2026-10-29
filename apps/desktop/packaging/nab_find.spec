# PyInstaller spec for NAB Find (one-folder build).
#   venv-desktop\Scripts\pyinstaller.exe apps\desktop\packaging\nab_find.spec --noconfirm ^
#       --distpath apps\desktop\dist --workpath apps\desktop\build
# Needs models\desktop (tools\export_clip.py + the YOLO 384x640 ONNX file) before building.
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

HERE = Path(SPECPATH)
DESKTOP = HERE.parent
ROOT = DESKTOP.parents[1]
MODELS = ROOT / "models" / "desktop"
# OpenVINO IR with fp16-stored weights (tools/convert_ir.py)
MODEL_FILES = ["clip_image.xml", "clip_image.bin", "clip_text.xml", "clip_text.bin", "yolo11n_384x640.xml",
               "yolo11n_384x640.bin", "clip.json", "bpe_simple_vocab_16e6.txt.gz"]
# OpenVINO parts the app never uses: NPU plugin (~104 MB), frontends other than IR, link libraries,
# C headers, developer tools.
OV_DROP = ("npu", "tensorflow", "pytorch", "paddle", "jax", "gguf", "onnx_frontend", "hetero", "auto_batch",
           "_debug", ".lib", "openvino/include/", "openvino/tools/")

ov_datas, ov_bins, ov_hidden = collect_all("openvino")
datas = [(str(DESKTOP / "web"), "web")] + [(str(MODELS / f), "models") for f in MODEL_FILES]
datas += ov_datas + collect_data_files("imageio_ffmpeg") + collect_data_files("ftfy")

a = Analysis(
    [str(HERE / "launcher.py")],
    pathex=[str(DESKTOP), str(ROOT)],
    binaries=ov_bins,
    datas=datas,
    hiddenimports=ov_hidden + ["nab_find.server", "uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto",
                               "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"],
    excludes=["onnxruntime", "onnx", "torch", "torchvision", "open_clip", "faiss", "matplotlib", "polars", "pytest", "hypothesis",
              "playwright", "tkinter", "IPython", "pandas", "scipy", "transformers", "huggingface_hub"],
    noarchive=False,
)


def keep(entry) -> bool:
    name = entry[0].replace(chr(92), "/").lower()  # normalise Windows separators
    return not ("openvino" in name and any(d in name for d in OV_DROP))


a.binaries = [b for b in a.binaries if keep(b)]
a.datas = [d for d in a.datas if keep(d)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="NAB Find", icon=str(HERE / "nab_find.ico"),
          console=False, version=None)
coll = COLLECT(exe, a.binaries, a.datas, name="NAB Find")
