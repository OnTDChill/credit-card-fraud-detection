#!/usr/bin/env python3
"""
MoMo DSS Dashboard Launcher
============================
Simple launcher for the CEO Decision Support System dashboard.

Usage:
    python run_dashboard.py

Dashboard will be available at: http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch Streamlit dashboard on port 8501."""
    project_root = Path(__file__).resolve().parent
    app_path = project_root / "streamlit_app" / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Dashboard app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching MoMo DSS Dashboard...")
    print(f"📊 Project root: {project_root}")
    print(f"🎯 Dashboard URL: http://localhost:8501")
    print("-" * 50)
    
    try:
        # Change to streamlit_app directory for proper module imports
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--server.headless=true",
        ]
        
        subprocess.run(cmd, cwd=str(project_root / "streamlit_app"))
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
