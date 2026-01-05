# Lightweight psutil stub for test environments without psutil installed.
# Provides only the APIs used by performance_monitor.py
class _MemInfo:
    def __init__(self, rss: int = 0):
        self.rss = rss

class _Process:
    def __init__(self, pid=None):
        self._pid = pid
    def cpu_percent(self):
        return 0.0
    def memory_info(self):
        return _MemInfo(0)

def Process(pid):
    return _Process(pid)
