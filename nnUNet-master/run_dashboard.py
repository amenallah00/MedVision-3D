"""
MedVision 3D - Dashboard Launcher
Launches the FastAPI backend serving the React dashboard and opens it in the browser.

Project: MedVision 3D
Maintainer: Amen Allah Ben Abdrabah
"""

import sys
import webbrowser
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = ROOT_DIR / ".venv" / "Scripts" / "python.exe"

def main():
    print("=" * 70)
    print("  MedVision 3D - AI Medical Image Segmentation Dashboard")
    print("  Maintainer: Amen Allah Ben Abdrabah")
    print("=" * 70)
    print("Starting server on http://127.0.0.1:8000 ...")

    # Open browser after a brief delay
    webbrowser.open("http://127.0.0.1:8000")

    # Run uvicorn
    cmd = [
        str(PYTHON_EXE if PYTHON_EXE.exists() else sys.executable),
        "-m", "uvicorn",
        "server:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    subprocess.run(cmd, cwd=str(ROOT_DIR))

if __name__ == "__main__":
    main()
