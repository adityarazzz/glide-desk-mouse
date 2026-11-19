"""Names of the video cameras, in the order OpenCV's Media Foundation backend numbers them.

Windows numbers only the cameras that work right now, so a camera's number changes when another
one appears or fails (a laptop's own camera coming back turned the desk camera from 0 into 1, and
Glide quietly showed the laptop camera). Glide therefore remembers the desk camera by its device
id and looks its number up here each time it opens it.

OpenCV opens MSMF camera i as the i-th entry of MFEnumDeviceSources(VIDCAP), so this list uses
the same call (plain ctypes COM, no extra packages).
"""
import ctypes
import re
import sys
from ctypes import POINTER, byref, c_uint32, c_void_p, c_wchar_p


class GUID(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16), ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, s):
        h = s.replace("-", "")
        return cls(int(h[0:8], 16), int(h[8:12], 16), int(h[12:16], 16),
                   (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:32])))


SOURCE_TYPE = GUID.of("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
SOURCE_TYPE_VIDCAP = GUID.of("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
FRIENDLY_NAME = GUID.of("60d0e559-52f8-4fa2-bbce-acdb34a8ec01")
SYMBOLIC_LINK = GUID.of("58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f")
SET_GUID, GET_ALLOCATED_STRING, RELEASE = 24, 13, 2      # IMFAttributes / IUnknown vtable slots


def _method(obj, slot, restype, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[slot])


def cameras():
    """[{"index", "name", "id"}] in OpenCV MSMF order; [] where Media Foundation isn't available.
    "id" is the USB vendor/product (vid_xxxx&pid_xxxx) when the link has one, else the link."""
    if sys.platform != "win32":
        return []
    try:
        ole32, mfplat, mf = ctypes.windll.ole32, ctypes.windll.mfplat, ctypes.windll.mf
    except OSError:
        return []
    ole32.CoInitializeEx(None, 0)                         # fine if the thread already is initialised
    attrs = c_void_p()
    if mfplat.MFCreateAttributes(byref(attrs), 1) != 0:
        return []
    out = []
    try:
        _method(attrs, SET_GUID, ctypes.c_long, POINTER(GUID), POINTER(GUID))(attrs, byref(SOURCE_TYPE),
                                                                            byref(SOURCE_TYPE_VIDCAP))
        devs, n = POINTER(c_void_p)(), c_uint32()
        if mf.MFEnumDeviceSources(attrs, byref(devs), byref(n)) != 0:
            return []
        get = None
        for i in range(n.value):
            act = devs[i]
            get = _method(act, GET_ALLOCATED_STRING, ctypes.c_long, POINTER(GUID), POINTER(c_wchar_p), POINTER(c_uint32))

            def read(key):
                s, ln = c_wchar_p(), c_uint32()
                if get(act, byref(key), byref(s), byref(ln)) != 0:
                    return ""
                v = s.value or ""
                ole32.CoTaskMemFree(s)
                return v
            name, link = read(FRIENDLY_NAME), read(SYMBOLIC_LINK)
            m = re.search(r"vid_[0-9a-f]{4}&pid_[0-9a-f]{4}", link.lower())
            out.append({"index": i, "name": name or f"Camera {i}", "id": m.group(0) if m else link.lower()})
            _method(act, RELEASE, c_uint32)(act)
        ole32.CoTaskMemFree(devs)
    except (OSError, ValueError):
        return out
    finally:
        _method(attrs, RELEASE, c_uint32)(attrs)
    return out


def index_of(cam_id, cams=None):
    """The current number of the camera with this id, or None if it isn't plugged in."""
    if not cam_id:
        return None
    for c in (cameras() if cams is None else cams):
        if c["id"] == cam_id:
            return c["index"]
    return None


if __name__ == "__main__":
    for c in cameras():
        print(c)
