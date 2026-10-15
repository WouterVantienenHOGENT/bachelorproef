"""Install official P2PNet weights and compact copies of existing CSRNet runs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from model_catalog import MODEL_DIR

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "5c91a81ca062b1c7fd3db3ad1c55b1c21f0a7455"
URL = f"https://raw.githubusercontent.com/TencentYoutuResearch/CrowdCounting-P2PNet/{COMMIT}/weights/SHTechA.pth"
SHA256 = "506047732b128ff09efef18e94bfacbe35fcfef300e5e9eeeece259b0488c63f"
RUNS = [
    ("shanghaitech_part_b_gpu_512_b4_e100_seed42", "csrnet_finaal", "CSRNet · finale training", "100 epochs, geselecteerd checkpoint uit epoch 39."),
    ("shanghaitech_part_b_cpu_first", "csrnet_cpu_kalibratie", "CSRNet · CPU-kalibratie", "Eerste kalibratierun op CPU."),
    ("shanghaitech_part_b_gpu_calibration_512_b4", "csrnet_gpu_kalibratie", "CSRNet · GPU-kalibratie", "Eerste kalibratierun op GPU."),
    ("shanghaitech_part_b_gpu_512_b4_e1_seed42_finalcode", "csrnet_een_epoch", "CSRNet · één epoch", "Gecontroleerde training van één epoch."),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(directory=MODEL_DIR, p2pnet_source=None):
    directory = Path(directory)
    target = directory / "p2pnet/p2pnet_shanghaitech_a.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != SHA256:
            raise ValueError(f"Bestaand bestand heeft een andere hash; niet overschreven: {target}")
    else:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                if p2pnet_source:
                    with Path(p2pnet_source).open("rb") as source:
                        shutil.copyfileobj(source, stream)
                else:
                    with urlopen(URL, timeout=60) as source:
                        shutil.copyfileobj(source, stream)
                stream.flush()
                if digest(temporary) != SHA256:
                    raise ValueError("P2PNet-download heeft een onverwachte SHA-256-hash.")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    info = {"name": "P2PNet · ShanghaiTech A", "architecture": "p2pnet",
            "description": "Officieel voorgetraind model. Telt hoofdposities; geen lokale training.",
            "source": URL, "source_commit": COMMIT, "sha256": SHA256,
            "license": "Academic research only; crowd_counting/vendor/p2pnet/LICENSE"}
    if not target.with_suffix(".json").exists():
        target.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"P2PNet beschikbaar: {target}")
    import torch
    for run, filename, name, description in RUNS:
        source = ROOT / "bachproef/runs" / run / "best.pt"
        target = directory / "csrnet" / f"{filename}.pt"
        if target.exists() or not source.exists():
            continue
        # Existing local training files contain RNG and optimizer state as well.
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": checkpoint["model"], "epoch": checkpoint.get("epoch")}, target)
        target.with_suffix(".json").write_text(json.dumps({
            "name": name, "architecture": "csrnet", "description": description,
            "source": str(source.relative_to(ROOT)), "source_sha256": digest(source),
            "sha256": digest(target), "training_dataset": "ShanghaiTech Part B",
        }, indent=2, ensure_ascii=False) + "\n")
        print(f"CSRNet gewichten gekopieerd zonder optimizer: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=MODEL_DIR)
    parser.add_argument("--p2pnet-source", type=Path, help="Gebruik een reeds gedownload officieel checkpoint")
    args = parser.parse_args()
    install(args.directory, args.p2pnet_source)
