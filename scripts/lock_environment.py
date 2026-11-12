"""Record the actually installed dependency closure, excluding unrelated tools."""
from importlib import metadata
from pathlib import Path
import json
import platform

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


roots = ["torch", "numpy", "pillow", "requests", "flwr", "opacus", "onnx", "onnxruntime",
         "onnxscript", "psutil", "pandas", "matplotlib", "streamlit", "scikit-learn", "pytest", "pypdf"]
found = {}
pending = roots.copy()
while pending:
    name = canonicalize_name(pending.pop())
    if name in found:
        continue
    dist = metadata.distribution(name)
    found[name] = dist.version
    for text in dist.requires or []:
        req = Requirement(text)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            pending.append(req.name)
Path("requirements-lock.txt").write_text(
    "# Tested Windows/Python 3.11 dependency closure. CUDA torch requires its matching official wheel index.\n"
    + "\n".join(f"{name}=={value}" for name, value in sorted(found.items())) + "\n", encoding="utf-8")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/environment.json").write_text(json.dumps({"python": platform.python_version(),
    "platform": platform.platform(), "packages": found}, indent=2), encoding="utf-8")
print(f"Locked {len(found)} packages")
