from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import tyro

from world_model.pi05_wan_action_expert import (
    load_pi05_wan_prefix_cache_row,
    load_wan_pi05_action_expert_checkpoint,
    predict_denormalized_action_chunk,
)


@dataclasses.dataclass
class Args:
    checkpoint: str
    prefix_cache_row: str
    output_dir: str = "output/pi05_wan_action_expert_infer"
    sample_steps: int = 16
    device: str = "auto"
    flow_seed: int | None = 0
    zero_noise: bool = False


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_wan_action_mode(
    *,
    checkpoint_mode: str | None,
    row_mode: str | None,
    checkpoint_path: str,
    row_path: str,
) -> str | None:
    if checkpoint_mode is not None and row_mode is not None and checkpoint_mode != row_mode:
        raise ValueError(
            "wan_action_mode disagrees between checkpoint and prefix cache row: "
            f"{checkpoint_path} has {checkpoint_mode!r}, {row_path} has {row_mode!r}."
        )
    return checkpoint_mode if checkpoint_mode is not None else row_mode


def main(args: Args) -> dict[str, object]:
    device = _resolve_device(args.device)
    loaded = load_wan_pi05_action_expert_checkpoint(args.checkpoint, device=device)
    row = load_pi05_wan_prefix_cache_row(args.prefix_cache_row)
    wan_action_mode = _resolve_wan_action_mode(
        checkpoint_mode=loaded.wan_action_mode,
        row_mode=row.get("wan_action_mode"),
        checkpoint_path=args.checkpoint,
        row_path=args.prefix_cache_row,
    )

    noise = None
    generator = None
    noise_mode = "random"
    if args.zero_noise:
        noise = torch.zeros(
            loaded.model.action_horizon,
            loaded.model.action_dim,
            device=device,
            dtype=next(loaded.model.parameters()).dtype,
        )
        noise_mode = "zero"
    elif args.flow_seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.flow_seed)

    action_chunk = predict_denormalized_action_chunk(
        loaded,
        row["prefix_tokens"],
        row["state"],
        num_steps=args.sample_steps,
        noise=noise,
        generator=generator,
        tasks=row["task"],
    )

    output: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)),
        "prefix_cache_row": str(Path(args.prefix_cache_row)),
        "sample_steps": args.sample_steps,
        "flow_seed": args.flow_seed if noise_mode == "random" else None,
        "noise": noise_mode,
        "action_chunk": action_chunk.detach().cpu().tolist(),
    }
    if wan_action_mode is not None:
        output["wan_action_mode"] = wan_action_mode

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pi05_wan_action.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return output


if __name__ == "__main__":
    main(tyro.cli(Args))
