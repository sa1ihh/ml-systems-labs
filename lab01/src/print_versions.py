"""Print the Python version and the installed version of every pinned package.

Usage (from repo root):
    python lab01/src/print_versions.py > lab01/results/versions.txt
"""
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements.txt"


def pinned_packages():
    """Read package names (left of '==') from requirements.txt."""
    names = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split("==")[0])
    return names


def main():
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print()
    for name in pinned_packages():
        try:
            print(f"{name}=={version(name)}")
        except PackageNotFoundError:
            print(f"{name}: NOT INSTALLED")


if __name__ == "__main__":
    main()
