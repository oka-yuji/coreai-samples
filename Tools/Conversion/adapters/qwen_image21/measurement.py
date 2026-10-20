from contextlib import contextmanager
import ctypes
import sys
import time

from .evaluation import measurement


class TaskVMInfoV3(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("region_count", ctypes.c_int32), ("page_size", ctypes.c_int32)]
    _fields_ += [(name, ctypes.c_uint64) for name in (
        "resident_size", "resident_size_peak", "device", "device_peak", "internal", "internal_peak", "external",
        "external_peak", "reusable", "reusable_peak", "purgeable_volatile_pmap", "purgeable_volatile_resident",
        "purgeable_volatile_virtual", "compressed", "compressed_peak", "compressed_lifetime", "phys_footprint",
        "min_address", "max_address")]
    _fields_ += [(name, ctypes.c_int64) for name in (
        "ledger_phys_footprint_peak", "ledger_purgeable_nonvolatile", "ledger_purgeable_novolatile_compressed",
        "ledger_purgeable_volatile", "ledger_purgeable_volatile_compressed", "ledger_tag_network_nonvolatile",
        "ledger_tag_network_nonvolatile_compressed", "ledger_tag_network_volatile", "ledger_tag_network_volatile_compressed",
        "ledger_tag_media_footprint", "ledger_tag_media_footprint_compressed", "ledger_tag_media_nofootprint",
        "ledger_tag_media_nofootprint_compressed", "ledger_tag_graphics_footprint", "ledger_tag_graphics_footprint_compressed",
        "ledger_tag_graphics_nofootprint", "ledger_tag_graphics_nofootprint_compressed", "ledger_tag_neural_footprint",
        "ledger_tag_neural_footprint_compressed", "ledger_tag_neural_nofootprint", "ledger_tag_neural_nofootprint_compressed")]


def thermal_state():
    if sys.platform != "darwin":
        return {"value": None, "status": "unavailable"}
    try:
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        message = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        scalar = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
        process = message(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
        value = scalar(process, objc.sel_registerName(b"thermalState"))
        return {"value": ("nominal", "fair", "serious", "critical")[value], "status": "ok"}
    except (OSError, ValueError, IndexError):
        return {"value": None, "status": "failed"}


def memory_snapshot():
    scope = "task_info point observation; not a phase peak"
    values = {name: measurement(None, "unavailable", "B", scope) for name in
              ("rss", "footprint", "graphics_footprint", "kernel_lifetime_peak", "metal_allocated", "available_app")}
    values["available_app"]["scope"] = "iOS-only os_proc_available_memory; unavailable to this macOS host"
    values["metal_allocated"]["scope"] = "Core AI Python does not expose the delegate's MTLDevice allocation counter"
    values["kernel_lifetime_peak"]["scope"] = "kernel physical-footprint peak over this process lifetime; never reset per phase"
    if sys.platform == "darwin":
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        info = TaskVMInfoV3()
        count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
        library.task_info.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        library.task_info.restype = ctypes.c_int
        task = ctypes.c_uint32.in_dll(library, "mach_task_self_").value
        result = library.task_info(task, 22, ctypes.byref(info), ctypes.byref(count))
        observed = {"rss": info.resident_size, "footprint": info.phys_footprint,
                    "graphics_footprint": info.ledger_tag_graphics_footprint + info.ledger_tag_graphics_footprint_compressed,
                    "kernel_lifetime_peak": info.ledger_phys_footprint_peak}
        valid = result == 0 and count.value * 4 >= ctypes.sizeof(info)
        for name, value in observed.items():
            status = "ok" if valid and value >= 0 else "failed"
            values[name] = measurement(value if status == "ok" else None, status, "B", values[name]["scope"])
    return {"monotonic_ns": time.monotonic_ns(), "memory": values, "thermal": thermal_state()}


class PhaseRecorder:
    def __init__(self, capture=memory_snapshot, clock=time.monotonic_ns):
        self.capture = capture
        self.clock = clock
        self.started = clock()
        self.phases = []
        self.observations = []
        self.specialization = measurement(None, "unavailable", "s", "AIModel.load and first call may include specialization; not separately exposed")

    @contextmanager
    def phase(self, name):
        before = self.capture()
        start = self.clock()
        row = {"phase": name, "start_ns": start, "status": "running"}
        self.phases.append(row)
        try:
            yield
        except BaseException as error:
            row.update(status="failed", error_type=type(error).__name__)
            raise
        else:
            row["status"] = "completed"
        finally:
            end = self.clock()
            row.update(end_ns=end, seconds=(end - start) / 1e9)
            self.observations.extend([{"phase": name, "boundary": "before", **before},
                                      {"phase": name, "boundary": "after", **self.capture()}])

    def report(self):
        return {"format": 1, "clock": "monotonic_ns", "seconds": (self.clock() - self.started) / 1e9,
                "phases": self.phases, "observations": self.observations,
                "specialization_seconds": self.specialization,
                "scope": "host-observed phase intervals and point memory samples; no inferred phase peaks"}
