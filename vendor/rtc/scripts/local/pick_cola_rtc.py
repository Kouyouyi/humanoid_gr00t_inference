"""Prepare and serve local pick-cola and Mickey RTC model bundles."""

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    "quality": "pick-cola-rtc-quality-20261005",
    "noquality": "pick-cola-rtc-noquality-20261005",
    "old": "pick-cola-rtc-old10pct-newfull-20261004",
    "mickey": "pick_mickey_sofa_rtc_quality_bs64_8ep_final",
}
PROMPTS = {name: "pick up the cola" for name in MODELS}
PROMPTS["mickey"] = "pick up the mickey on the sofa"


def prepare_model(model_root: Path, name: str) -> Path:
    bundle = model_root / name
    if not (bundle / "config.json").is_file():
        bundle = bundle / name
    bundle = bundle.resolve()
    backbone = bundle / "shared/Cosmos-Reason2-2B"
    required = [
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
        "shared/Cosmos-Reason2-2B/model.safetensors",
    ]
    for filename in required:
        if not (bundle / filename).is_file():
            raise FileNotFoundError(bundle / filename)
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    for filename in set(index["weight_map"].values()):
        if not (bundle / filename).is_file():
            raise FileNotFoundError(bundle / filename)

    # Keep the delivered bundle and its checksums intact. Only small configs are
    # copied into this ignored local directory; weights remain on the SSD.
    local = REPO_ROOT / "models" / name
    local.mkdir(parents=True, exist_ok=True)
    for source in bundle.iterdir():
        target = local / source.name
        if source.name in ("config.json", "processor_config.json"):
            if target.is_symlink():
                raise ValueError(f"Refusing to write config through symlink: {target}")
            config = json.loads(source.read_text())
            if source.name == "config.json":
                config["model_name"] = str(backbone)
            else:
                config["processor_kwargs"]["model_name"] = str(backbone)
            target.write_text(json.dumps(config, indent=2) + "\n")
        elif target.is_symlink():
            if target.resolve() != source:
                raise ValueError(f"Existing link points to a different bundle: {target}")
        elif target.exists():
            raise FileExistsError(f"Expected a symlink: {target}")
        else:
            target.symlink_to(source, target_is_directory=source.is_dir())
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--model-root", type=Path, default=Path("/media/user/My_PSSD/202_models"))
    parser.add_argument("--mode", choices=("continuous", "grid"), default="continuous")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument(
        "--rtc-delay-steps",
        type=int,
        choices=range(1, 11),
        default=8,
        help="RTC control-step delay advertised to clients (default: 8)",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    local = prepare_model(args.model_root, MODELS[args.model])
    print(f"Model: {local}", flush=True)
    if args.prepare_only:
        return
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", NO_ALBUMENTATIONS_UPDATE="1")
    os.chdir(REPO_ROOT)
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "gr00t.eval.run_gr00t_server",
            "--model-path",
            str(local),
            "--embodiment-tag",
            "UNITREE_G1_SONIC",
            "--sonic-token-mode",
            args.mode,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--rtc-delay-steps",
            str(args.rtc_delay_steps),
        ],
    )


if __name__ == "__main__":
    main()
