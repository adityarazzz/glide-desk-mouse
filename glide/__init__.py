"""Glide P1 prototype: bare-hand desk mouse driven by one external camera."""
import os

# Media Foundation's hardware transforms make opening a camera take ~8 s; without them 0.7 s
# (measured on the ZEB LIVE PRO), same 30 fps. Must be set before cv2 opens a device.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

# NumPy's OpenBLAS commits a buffer per CPU thread when it loads: 363 MB of private memory on a
# 12-thread laptop (measured), for matrix products Glide does on 2x2 and 21x3 arrays. One thread
# is all it needs (NFR-05). Must be set before numpy is imported.
__version__ = "0.9.0"
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
