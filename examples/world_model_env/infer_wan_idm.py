from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch
import tyro

from world_model.config import DatasetConfig, DatasetSource, FutureFrameStrategy, Wan22Config
from world_model.data import (
    create_dataset,
    expected_wan_source_frame_offsets,
    validate_raw_wan_frame_delta,
    validate_wan_selected_frame_indices,
)
from world_model.train_lib import (
    create_flow_sample_noise,
    enforce_idm_frame_delta_contract,
    get_action_normalizer,
    get_state_normalizer,
    idm_history_kwargs,
    idm_uses_flow_matching,
    load_idm_checkpoint,
    normalize_state_for_idm,
    resolve_device,
)
from world_model.wan22 import Wan22FutureGenerator


@dataclasses.dataclass
class Args:
    idm_checkpoint: str
    wan_repo_dir: str
    wan_checkpoint_dir: str
    dataset_source: DatasetSource = "lerobot"
    repo_id: str = "brandonyang/metaworld_ml45"
    image_key: str = "corner4.image"
    output_dir: str = "output/wan_idm"
    sample_index: int = 0
    episodes: tuple[int, ...] | None = None
    max_samples: int | None = None
    frame_delta: int = 1
    task_prompt: str | None = None
    wan_task: str = "ti2v-5B"
    wan_size: str = "1280*704"
    wan_frame_num: int = 17
    wan_sample_steps: int | None = None
    wan_sample_shift: float | None = None
    wan_sample_guide_scale: float | None = None
    wan_offload_model: bool = False
    wan_convert_model_dtype: bool = False
    wan_t5_cpu: bool = False
    wan_python_executable: str = "python"
    wan_future_frame_strategy: FutureFrameStrategy = "first"
    idm_flow_seed: int | None = 0
    seed: int = 7
    device: str = "auto"


def main(args: Args) -> None:
    validate_raw_wan_frame_delta(args.frame_delta, context="Live Wan2.2 inference")
    device = resolve_device(args.device)
    idm, model_config = load_idm_checkpoint(args.idm_checkpoint, device)
    enforce_idm_frame_delta_contract(args.idm_checkpoint, args.frame_delta)
    action_normalizer = get_action_normalizer(idm, device)
    state_normalizer = get_state_normalizer(idm, device)
    if model_config.num_views != 1:
        raise ValueError(
            "Wan2.2 inference currently generates one selected camera view. "
            f"Train or load a single-view IDM checkpoint; got num_views={model_config.num_views}."
        )

    dataset = create_dataset(
        DatasetConfig(
            source=args.dataset_source,
            repo_id=args.repo_id,
            image_keys=(args.image_key,),
            frame_delta=args.frame_delta,
            num_future_frames=model_config.num_future_frames,
            action_horizon=model_config.action_horizon,
            image_size=model_config.image_size,
            max_samples=args.max_samples,
            episodes=args.episodes,
            task_vocab_size=model_config.task_vocab_size,
            idm_history_length=model_config.idm_history_length,
            seed=args.seed,
        )
    )
    item = dataset[args.sample_index]
    if args.task_prompt is None:
        if not hasattr(dataset, "task_text"):
            raise ValueError("--task-prompt is required when the dataset does not expose task_text().")
        task_prompt = dataset.task_text(args.sample_index)
    else:
        task_prompt = args.task_prompt

    wan = Wan22FutureGenerator(
        Wan22Config(
            repo_dir=args.wan_repo_dir,
            checkpoint_dir=args.wan_checkpoint_dir,
            task=args.wan_task,
            size=args.wan_size,
            frame_num=args.wan_frame_num,
            sample_steps=args.wan_sample_steps,
            sample_shift=args.wan_sample_shift,
            sample_guide_scale=args.wan_sample_guide_scale,
            offload_model=args.wan_offload_model,
            convert_model_dtype=args.wan_convert_model_dtype,
            t5_cpu=args.wan_t5_cpu,
            base_seed=args.seed,
            python_executable=args.wan_python_executable,
            frame_delta=args.frame_delta,
            future_frame_strategy=args.wan_future_frame_strategy,
        )
    )
    output_dir = Path(args.output_dir)
    wan_result = wan.generate_future_stack(
        item["current_images"],
        task_text=task_prompt,
        output_dir=output_dir,
        image_size=model_config.image_size,
        num_future_frames=model_config.num_future_frames,
        seed=args.seed,
    )
    validate_wan_selected_frame_indices(
        wan_result.selected_frame_indices,
        frame_delta=args.frame_delta,
        num_future_frames=model_config.num_future_frames,
        strategy=args.wan_future_frame_strategy,
        context="Live Wan2.2 inference",
    )

    idm.eval()
    with torch.no_grad():
        current_images = item["current_images"].unsqueeze(0).to(device)
        future_images = wan_result.future_images.unsqueeze(0).to(device)
        state = item["state"].unsqueeze(0).to(device)
        task_id = item["task_id"].unsqueeze(0).to(device)
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in item.items()
            if key in {"prev_state_history", "prev_action_history", "history_mask"}
        }
        sample_noise = None
        if idm_uses_flow_matching(idm):
            generator = None
            if args.idm_flow_seed is not None:
                generator = torch.Generator(device=device).manual_seed(args.idm_flow_seed)
            sample_noise = create_flow_sample_noise(
                idm,
                batch_size=1,
                device=device,
                dtype=current_images.dtype,
                generator=generator,
            )
        model_action = idm(
            current_images,
            future_images,
            normalize_state_for_idm(idm, state, state_normalizer),
            task_id,
            sample_noise=sample_noise,
            **idm_history_kwargs(
                batch,
                idm=idm,
                action_normalizer=action_normalizer,
                state_normalizer=state_normalizer,
            ),
        )[0]
        action = (
            model_action if action_normalizer is None else action_normalizer.denormalize(model_action.unsqueeze(0))[0]
        )

    output = {
        "prompt": wan_result.prompt,
        "seed": wan_result.seed,
        "idm_flow_seed": args.idm_flow_seed if idm_uses_flow_matching(idm) else None,
        "future_frame_strategy": args.wan_future_frame_strategy,
        "selected_frame_indices": list(wan_result.selected_frame_indices),
        "dataset_frame_delta": args.frame_delta,
        "source_frame_offsets": expected_wan_source_frame_offsets(
            args.frame_delta,
            model_config.num_future_frames,
        ),
        "total_video_frames": wan_result.total_video_frames,
        "input_image_path": str(wan_result.input_image_path),
        "wan_video_path": str(wan_result.video_path),
        "action_chunk": action.detach().cpu().tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "wan_idm_action.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main(tyro.cli(Args))
