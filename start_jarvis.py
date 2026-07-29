import subprocess
import sys
import os
import time
import webbrowser

def main():
    print("=" * 60)
    print("      STARTING JARVIS AI BRAIN - PC CONTROL AGENT      ")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, "backend")
    frontend_dir = os.path.join(base_dir, "frontend")

    print("\n[1/3] Starting FastAPI Core Backend Server (port 8000)...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=backend_dir
    )

    time.sleep(2)

    print("\n[2/3] Starting Futuristic Vite Command Center Dashboard (port 5173)...")
    frontend_process = subprocess.Popen(
        "npm run dev",
        shell=True,
        cwd=frontend_dir
    )

    time.sleep(3)

    dashboard_url = "http://localhost:5173"
    print(f"\n[3/3] Launching JARVIS Dashboard: {dashboard_url}")
    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("   JARVIS AI Brain is running at http://localhost:5173 !")
    print("=" * 60)

    try:
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        print("\nStopping JARVIS AI Brain servers...")
        backend_process.terminate()
        frontend_process.terminate()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
