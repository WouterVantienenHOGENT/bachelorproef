"""Discover locally installed weights without loading them into memory."""

from dataclasses import dataclass
import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
ARCHITECTURES = ("csrnet", "p2pnet")


@dataclass(frozen=True)
class ModelEntry:
    path: Path
    name: str
    architecture: str
    description: str = ""


def discover_models(directory=MODEL_DIR):
    entries, errors = [], []
    root = Path(directory)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".pt", ".pth"}:
            continue
        metadata = path.with_suffix(".json")
        try:
            info = json.loads(metadata.read_text()) if metadata.exists() else {}
            if not isinstance(info, dict):
                raise ValueError("metadata moet een JSON-object zijn")
            architecture = info.get("architecture", path.parent.name.lower())
            if architecture not in ARCHITECTURES:
                raise ValueError("plaats gewichten in csrnet/ of p2pnet/, of voeg architecture toe aan de JSON")
            name = info.get("name", path.stem.replace("_", " ").replace("-", " "))
            description = info.get("description", "")
            if not isinstance(name, str) or not name.strip() or not isinstance(description, str):
                raise ValueError("name en description moeten tekst bevatten")
            entries.append(ModelEntry(path, name, architecture, description))
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    return sorted(entries, key=lambda entry: (entry.architecture, entry.name)), errors
