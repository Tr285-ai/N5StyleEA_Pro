# This sitecustomize ensures tests run even if optional system packages like psutil are missing.
# It is imported automatically by Python at startup if present on sys.path.
import sys

# Provide a lightweight psutil stub if psutil is not installed.
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - only used in test environments
    class _MemInfo:
        def __init__(self, rss: int = 0):
            self.rss = rss

    class _Process:
        def __init__(self, _pid=None):
            pass
        def cpu_percent(self):
            return 0.0
        def memory_info(self):
            return _MemInfo(0)

    class _PsutilStub:
        def Process(self, pid):
            return _Process(pid)

    sys.modules['psutil'] = _PsutilStub()  # type: ignore
