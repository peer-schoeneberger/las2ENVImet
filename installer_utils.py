# installer_utils.py
# Utilities for automatically installing missing Python packages

import sys
import subprocess
import os
from pathlib import Path
import platform
from typing import List, Tuple

def locate_qgis_python() -> Path:
    # Locate python interpreter
    if platform.system() == "Windows":
        qgis_bin_dir = Path(sys.exec_prefix).parent / "bin"
        candidates = [
            qgis_bin_dir / "python.exe",
            Path(sys.exec_prefix) / "python.exe",
            Path(sys.executable).with_name("python.exe"),
            Path(sys.executable),
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        for cand in [
            qgis_bin_dir / "pythonw.exe",
            Path(sys.exec_prefix) / "pythonw.exe",
        ]:
            if cand.exists():
                return cand
    else:
        # Linux/Mac
        if sys.executable:
            exe = Path(sys.executable)
            if exe.exists():
                return exe
        for cand in ["/usr/bin/python3", "/usr/local/bin/python3"]:
            p = Path(cand)
            if p.exists():
                return p

    raise RuntimeError("Could not find Python interpreter for QGIS.")

def ensure_pip() -> bool:
    try:
        python_exe = locate_qgis_python()
        result = subprocess.run(
            [str(python_exe), "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def install_packages_sync(packages: List[str], upgrade: bool = False, user: bool = True) -> Tuple[bool, str]:
    try:
        python_exe = locate_qgis_python()
    except RuntimeError as e:
        return False, f"Python interpreter not found: {e}"

    cmd = [str(python_exe), "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    if user:
        cmd.append("--user")
    cmd.extend(packages)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            check=False
        )
        if result.returncode == 0:
            return True, f"Installation successful.\n\nOutput:\n{result.stdout}"
        else:
            return False, f"Installation failed (Code {result.returncode}).\n\nOutput:\n{result.stderr}\n\nDefault output:\n{result.stdout}"
    except subprocess.TimeoutExpired:
        return False, "Timed out after 5 minutes"
    except Exception as e:
        return False, f"Unexpected error: {e}"