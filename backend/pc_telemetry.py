import psutil
import platform
import os
import sys
import time
from typing import Dict, Any, List

# Windows-specific import for active window detection
try:
    import win32gui
    import win32process
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

class PCTelemetry:
    """Collects real-time hardware telemetry, process monitoring, active window details, and OS specs."""
    
    @staticmethod
    def get_system_specs() -> Dict[str, Any]:
        """Returns static & initial system hardware information."""
        specs = {
            "os_name": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_total_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "hostname": platform.node(),
            "python_version": sys.version.split()[0]
        }
        
        # Disk partitions
        disks = []
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disks.append({
                    "device": partition.device,
                    "mountpoint": partition.mountpoint,
                    "fstype": partition.fstype,
                    "total_gb": round(usage.total / (1024 ** 3), 2),
                    "free_gb": round(usage.free / (1024 ** 3), 2),
                    "used_percent": usage.percent
                })
            except Exception:
                pass
        specs["disks"] = disks
        return specs

    @staticmethod
    def get_live_metrics() -> Dict[str, Any]:
        """Returns real-time CPU, RAM, Disk, Active Window, and Top Process info."""
        mem = psutil.virtual_memory()
        cpu_per_core = psutil.cpu_percent(percpu=True, interval=None)
        cpu_overall = psutil.cpu_percent(interval=None)
        
        # Active foreground window info
        active_window_info = PCTelemetry.get_active_window()
        
        # Battery info if available
        battery_info = None
        try:
            bat = psutil.sensors_battery()
            if bat:
                battery_info = {
                    "percent": bat.percent,
                    "power_plugged": bat.power_plugged,
                    "time_left_secs": bat.secsleft
                }
        except Exception:
            pass

        return {
            "timestamp": time.time(),
            "cpu_percent": cpu_overall,
            "cpu_per_core": cpu_per_core,
            "ram_used_gb": round(mem.used / (1024 ** 3), 2),
            "ram_percent": mem.percent,
            "ram_available_gb": round(mem.available / (1024 ** 3), 2),
            "active_window": active_window_info,
            "battery": battery_info,
            "top_processes": PCTelemetry.get_top_processes(limit=5)
        }

    @staticmethod
    def get_active_window() -> Dict[str, Any]:
        """Gets title, process name, PID, and geometry of current active window."""
        if not HAS_WIN32:
            return {"title": "Unknown (pywin32 not loaded)", "pid": 0, "app": "unknown"}
        
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return {"title": "Desktop / None", "pid": 0, "app": "explorer.exe"}
            
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            app_name = "Unknown"
            try:
                proc = psutil.Process(pid)
                app_name = proc.name()
            except Exception:
                pass
                
            rect = win32gui.GetWindowRect(hwnd)
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]

            return {
                "title": title or "Untitled Window",
                "pid": pid,
                "app": app_name,
                "x": rect[0],
                "y": rect[1],
                "width": width,
                "height": height
            }
        except Exception as e:
            return {"title": "Desktop", "pid": 0, "app": "explorer.exe", "error": str(e)}

    @staticmethod
    def get_top_processes(limit: int = 8) -> List[Dict[str, Any]]:
        """Returns top CPU/RAM consuming processes."""
        procs = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                info = proc.info
                if info['name'] and info['name'] not in ['System Idle Process']:
                    procs.append({
                        'pid': info['pid'],
                        'name': info['name'],
                        'cpu': round(info['cpu_percent'] or 0.0, 1),
                        'mem': round(info['memory_percent'] or 0.0, 1)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Sort by CPU usage descending
        procs.sort(key=lambda x: x['cpu'], reverse=True)
        return procs[:limit]
