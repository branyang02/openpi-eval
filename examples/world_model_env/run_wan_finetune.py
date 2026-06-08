from __future__ import annotations

import dataclasses
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal

import tyro

from prepare_wan_finetune import Args as PrepareArgs
from prepare_wan_finetune import build_command, validate_dataset
from world_model.diffsynth_wan import validate_local_wan_checkpoint

TrainMode = Literal["lora", "full"]


@dataclasses.dataclass
class Args:
    dataset_dir: str
    diffsynth_repo_dir: str
    checkpoint_dir: str | None = None
    preflight_output_dir: str = "output/wan_finetune_preflight"
    output_path: str = "./models/train/Wan2.2-TI2V-5B_lora"
    mode: TrainMode = "lora"
    height: int = 480
    width: int = 832
    num_frames: int = 17
    dataset_repeat: int = 100
    learning_rate: float | None = None
    num_epochs: int | None = None
    lora_rank: int = 32
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B"
    model_paths: tuple[str, ...] | None = None
    model_paths_json: str | None = None
    tokenizer_path: str | None = None
    diffsynth_train_script: str = "examples/wanvideo/model_training/train.py"
    metadata_filename: str = "metadata.csv"
    data_file_keys: str = "video"
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True
    enable_model_cpu_offload: bool = False
    enable_optimizer_cpu_offload: bool = False
    accelerate_config_file: str | None = None
    accelerate_num_processes: int | None = None
    validate_videos: bool = True
    validate_diffsynth_import: bool = True
    run: bool = False


def resolve_path_text(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def make_prepare_args(args: Args, checkpoint_summary: dict[str, object] | None = None) -> PrepareArgs:
    model_paths_json = args.model_paths_json
    tokenizer_path = args.tokenizer_path
    if checkpoint_summary is not None:
        if args.model_paths is None and model_paths_json is None:
            model_paths_json = json.dumps(checkpoint_summary["model_paths"])
        if tokenizer_path is None:
            tokenizer_path = str(checkpoint_summary["tokenizer_path"])
    return PrepareArgs(
        dataset_dir=args.dataset_dir,
        output_path=resolve_path_text(args.output_path),
        mode=args.mode,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        dataset_repeat=args.dataset_repeat,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        model_id=args.model_id,
        model_paths=args.model_paths,
        model_paths_json=model_paths_json,
        tokenizer_path=tokenizer_path,
        diffsynth_train_script=args.diffsynth_train_script,
        metadata_filename=args.metadata_filename,
        data_file_keys=args.data_file_keys,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        enable_optimizer_cpu_offload=args.enable_optimizer_cpu_offload,
        accelerate_config_file=args.accelerate_config_file,
        accelerate_num_processes=args.accelerate_num_processes,
        validate_videos=args.validate_videos,
    )


def validate_diffsynth_repo(repo_dir: Path, train_script: str) -> dict[str, str | bool]:
    if not repo_dir.exists():
        raise FileNotFoundError(f"DiffSynth-Studio repo not found: {repo_dir}")
    train_script_path = repo_dir / train_script
    if not train_script_path.exists():
        raise FileNotFoundError(f"DiffSynth training script not found: {train_script_path}")
    if not (repo_dir / "diffsynth").exists():
        raise FileNotFoundError(f"DiffSynth package directory not found: {repo_dir / 'diffsynth'}")
    revision = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_dir,
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "repo_dir": str(repo_dir),
        "train_script": train_script,
        "train_script_exists": True,
        "git_revision": revision,
    }


def validate_checkpoint_dir(checkpoint_dir: str | None) -> dict[str, object] | None:
    if checkpoint_dir is None:
        return None
    return validate_local_wan_checkpoint(checkpoint_dir)


def validate_diffsynth_imports(repo_dir: Path) -> dict[str, object]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo_dir) if existing_pythonpath is None else f"{repo_dir}{os.pathsep}{existing_pythonpath}"
    code = (
        "import accelerate, diffsynth, peft, torch\n"
        "print({'torch': torch.__version__, 'cuda': torch.version.cuda})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "DiffSynth import preflight failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return {
        "python": sys.executable,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
    }


def command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_preflight(args: Args) -> dict[str, object]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    diffsynth_repo_dir = Path(args.diffsynth_repo_dir)
    checkpoint_summary = validate_checkpoint_dir(args.checkpoint_dir)
    prepare_args = make_prepare_args(args, checkpoint_summary)
    command = build_command(prepare_args, dataset_dir)
    preflight = {
        "dataset": validate_dataset(dataset_dir, args.metadata_filename, args.num_frames, args.validate_videos),
        "diffsynth": validate_diffsynth_repo(diffsynth_repo_dir, args.diffsynth_train_script),
        "checkpoint": checkpoint_summary,
        "command": command,
        "command_text": command_text(command),
        "cwd": str(diffsynth_repo_dir),
        "run_requested": args.run,
        "mode": args.mode,
    }
    if args.validate_diffsynth_import:
        preflight["python_imports"] = validate_diffsynth_imports(diffsynth_repo_dir)
    return preflight


def main(args: Args) -> None:
    preflight = build_preflight(args)
    preflight_output_dir = Path(args.preflight_output_dir)
    preflight_output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = preflight_output_dir / "wan_finetune_preflight.json"
    preflight_path.write_text(json.dumps(preflight, indent=2) + "\n")

    if args.run:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(args.diffsynth_repo_dir)
            if existing_pythonpath is None
            else f"{args.diffsynth_repo_dir}{os.pathsep}{existing_pythonpath}"
        )
        subprocess.run(preflight["command"], cwd=preflight["cwd"], env=env, check=True)

    print(json.dumps({"preflight": str(preflight_path), "run": args.run}, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
