"""Validate stable AXIS SemVer tags and both package version declarations."""
from pathlib import Path
import re
import sys
import tomllib

root = Path(__file__).resolve().parents[2]
version = tomllib.loads((root / "libs/axis/pyproject.toml").read_text())["project"]["version"]
semver = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
if not re.fullmatch(semver, version):
    sys.exit(f"AXIS version must be stable MAJOR.MINOR.PATCH: {version}")
cmake = (root / "libs/axis/CMakeLists.txt").read_text()
match = re.search(r"project\(\s*AXIS\s+VERSION\s+(\S+)", cmake)
if not match or match[1] != version:
    sys.exit("AXIS CMake and Python versions must match")
if len(sys.argv) > 1 and sys.argv[1] != f"axis-v{version}":
    sys.exit(f"Release tag must be axis-v{version}")
print(version)
