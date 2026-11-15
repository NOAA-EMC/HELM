"""Execute workers locally or in an existing Docker container with a shared mount."""

import json
import subprocess
from pathlib import Path


class NativeExecution:
    """Translate shared paths while leaving all measured work in native workers."""

    def __init__(self, container=None, host_root=None, container_root="/out"):
        if container and not host_root:
            raise ValueError("--container requires --host-root for the shared writable mount")
        self.container = container
        self.host_root = Path(host_root).resolve() if host_root else None
        self.container_root = Path(container_root)

    def translate(self, argument):
        path = Path(str(argument))
        if path.is_absolute():
            path = path.resolve()
        if self.container and path.is_absolute() and path.is_relative_to(self.host_root):
            return str(self.container_root / path.relative_to(self.host_root))
        return str(argument)

    def command(self, arguments, environment):
        if not self.container:
            return list(map(str, arguments))
        command = ["docker", "exec"]
        # docker exec does not inherit the caller's OpenMP settings automatically.
        for key in ("OMP_NUM_THREADS", "KOKKOS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"):
            if key in environment:
                command += ["--env", f"{key}={environment[key]}"]
        return command + [self.container] + [self.translate(value) for value in arguments]

    def provenance(self):
        if not self.container:
            return {"transport": "local"}
        result = subprocess.run(
            ["docker", "inspect", self.container], capture_output=True, text=True, check=True
        )
        details = json.loads(result.stdout)[0]
        expected = (str(self.host_root), str(self.container_root))
        mounts = {
            (str(Path(item["Source"]).resolve()), item["Destination"]) for item in details["Mounts"]
        }
        if expected not in mounts:
            raise ValueError(f"Container lacks declared shared mount: {expected}")
        return {
            "transport": "docker exec",
            "container": self.container,
            "image_id": details["Image"],
            "image_name": details["Config"]["Image"],
            "host_root": str(self.host_root),
            "container_root": str(self.container_root),
        }
