from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from cache_future_rollouts import Args as CacheArgs
from cache_future_rollouts import main as cache_main
from validate_wan_lora import (
    RANK_KEYS,
    Args,
    RankArgs,
    enforce_idm_frame_delta_contract,
    expected_selected_frame_indices,
    load_idm_training_frame_delta,
    rank_wan_lora_checkpoints,
    validate_cache_sample_identity,
    validate_cache_temporal_contract,
    validate_inputs,
    validate_ranking_inputs,
)
from world_model.config import DatasetConfig, TrainConfig
from world_model.data import expected_wan_source_frame_offsets
from world_model.train_lib import run_idm_training


def write_fake_wan_checkpoint(checkpoint_dir) -> None:
    for relative in [
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "google/umt5-xxl/spiece.model",
        "google/umt5-xxl/tokenizer.json",
        "google/umt5-xxl/tokenizer_config.json",
    ]:
        path = checkpoint_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stub\n")


def test_validate_wan_lora_validates_inputs(tmp_path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    lora_path = tmp_path / "epoch-0.safetensors"
    input_image = tmp_path / "input.png"
    (repo_dir / "diffsynth").mkdir(parents=True)
    write_fake_wan_checkpoint(checkpoint_dir)
    lora_path.write_text("stub\n")
    Image.new("RGB", (16, 16)).save(input_image)

    summary = validate_inputs(
        Args(
            diffsynth_repo_dir=str(repo_dir),
            checkpoint_dir=str(checkpoint_dir),
            lora_path=str(lora_path),
            input_image=str(input_image),
            prompt="Robot manipulation in MetaWorld.",
        )
    )

    assert summary["lora_path"] == str(lora_path)
    assert summary["input_image"] == str(input_image)
    assert summary["checkpoint"]["tokenizer_path"] == str(checkpoint_dir / "google" / "umt5-xxl")


def test_validate_wan_lora_rejects_missing_lora(tmp_path) -> None:
    repo_dir = tmp_path / "DiffSynth-Studio"
    checkpoint_dir = tmp_path / "Wan2.2-TI2V-5B"
    input_image = tmp_path / "input.png"
    (repo_dir / "diffsynth").mkdir(parents=True)
    write_fake_wan_checkpoint(checkpoint_dir)
    Image.new("RGB", (16, 16)).save(input_image)

    with pytest.raises(FileNotFoundError, match="LoRA checkpoint"):
        validate_inputs(
            Args(
                diffsynth_repo_dir=str(repo_dir),
                checkpoint_dir=str(checkpoint_dir),
                lora_path=str(tmp_path / "missing.safetensors"),
                input_image=str(input_image),
                prompt="Robot manipulation in MetaWorld.",
            )
        )


def build_dataset_future_cache(
    output_dir: Path,
    *,
    samples: int,
    num_future_frames: int = 4,
    action_horizon: int = 8,
    image_size: int = 32,
    frame_delta: int = 1,
    seed: int = 7,
) -> Path:
    """Build a cache in the same on-disk format produced for Wan LoRA futures.

    A ``dataset_future`` cache stores the ground-truth futures, so it is a stable
    stand-in for a LoRA cache without requiring a GPU or DiffSynth.
    """

    cache_main(
        CacheArgs(
            future_source="dataset_future",
            dataset_source="synthetic",
            output_dir=str(output_dir),
            max_samples=samples,
            synthetic_samples=samples,
            image_size=image_size,
            frame_delta=frame_delta,
            num_future_frames=num_future_frames,
            action_horizon=action_horizon,
            seed=seed,
        )
    )
    return Path(output_dir)


def train_tiny_idm(
    output_dir: Path,
    *,
    samples: int,
    num_future_frames: int = 4,
    action_horizon: int = 8,
    image_size: int = 32,
    frame_delta: int = 1,
    seed: int = 11,
) -> Path:
    run_idm_training(
        TrainConfig(
            dataset=DatasetConfig(
                source="synthetic",
                image_keys=("corner4.image",),
                image_size=image_size,
                frame_delta=frame_delta,
                max_samples=samples,
                synthetic_samples=samples,
                num_future_frames=num_future_frames,
                action_horizon=action_horizon,
                seed=seed,
            ),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
            device="cpu",
            seed=seed,
        )
    )
    return Path(output_dir) / "idm_checkpoint.pt"


def zero_out_cache_futures(cache_dir: Path) -> None:
    rows = [json.loads(line) for line in (cache_dir / "manifest.jsonl").read_text().splitlines() if line.strip()]
    for row in rows:
        future_path = cache_dir / row["future_tensor"]
        future = torch.load(future_path, map_location="cpu", weights_only=False)
        torch.save(torch.zeros_like(future), future_path)


def make_rank_args(idm_checkpoint: Path, cache_dirs, output_dir: Path, **overrides) -> RankArgs:
    defaults = dict(
        idm_checkpoint=str(idm_checkpoint),
        cache_dirs=tuple(str(path) for path in cache_dirs),
        dataset_source="synthetic",
        output_dir=str(output_dir),
        max_samples=6,
        synthetic_samples=6,
        frame_delta=1,
        seed=7,
        batch_size=4,
        device="cpu",
        visual_samples=2,
        visual_tile_size=24,
    )
    defaults.update(overrides)
    return RankArgs(**defaults)


def test_rank_key_idm_decodability_gap_ranks_by_magnitude() -> None:
    # idm_decodability_gap = idm_mse(generated) - idm_mse(ground_truth). The most faithful
    # checkpoint is the one whose generated futures are *as* action-decodable as the real
    # futures (gap closest to zero), so the ranking key is the gap's MAGNITUDE. A checkpoint
    # whose futures are artificially easier to decode (negative gap) must not rank above one
    # that matches the ground-truth reference.
    faithful = {"label": "faithful", "idm_decodability_gap": 0.0, "idm": {"idm_mse": 0.50}}
    too_easy = {"label": "too_easy", "idm_decodability_gap": -0.30, "idm": {"idm_mse": 0.20}}
    too_hard = {"label": "too_hard", "idm_decodability_gap": 0.10, "idm": {"idm_mse": 0.60}}

    gap_key = RANK_KEYS["idm_decodability_gap"]
    assert gap_key(faithful) == 0.0
    assert gap_key(too_easy) == pytest.approx(0.30)
    assert gap_key(too_hard) == pytest.approx(0.10)

    by_gap = [run["label"] for run in sorted([too_easy, faithful, too_hard], key=gap_key)]
    assert by_gap == ["faithful", "too_hard", "too_easy"]

    # Guard against regressing to the signed gap: it would rank the artificially-easy checkpoint
    # first and is anyway redundant with idm_mse (the ground-truth idm_mse is a constant offset,
    # so signed-gap order is identical to idm_mse order).
    by_signed = [
        run["label"] for run in sorted([too_easy, faithful, too_hard], key=lambda run: run["idm_decodability_gap"])
    ]
    by_idm_mse = [run["label"] for run in sorted([too_easy, faithful, too_hard], key=RANK_KEYS["idm_mse"])]
    assert by_signed == ["too_easy", "faithful", "too_hard"]
    assert by_signed == by_idm_mse
    assert by_gap != by_signed


def test_rank_wan_lora_checkpoints_ranks_by_pixel_and_idm(tmp_path) -> None:
    good_cache = build_dataset_future_cache(tmp_path / "good_cache", samples=6)
    bad_cache = build_dataset_future_cache(tmp_path / "bad_cache", samples=6)
    zero_out_cache_futures(bad_cache)
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    summary = rank_wan_lora_checkpoints(
        make_rank_args(
            idm_checkpoint,
            (good_cache, bad_cache),
            tmp_path / "ranking",
            labels=("good", "bad"),
        )
    )

    runs = {run["label"]: run for run in summary["runs"]}

    # Pixel future metrics are deterministic: ground-truth futures match exactly.
    assert runs["good"]["pixel"]["future_mse"] == 0.0
    assert runs["bad"]["pixel"]["future_mse"] > 0.0
    assert runs["good"]["num_samples"] == 6
    assert summary["ground_truth_reference"]["num_samples"] == 6

    # A cache holding the exact ground-truth futures must decode actions identically
    # to the ground-truth reference, so the action-decodability gap is zero.
    gt_idm_mse = summary["ground_truth_reference"]["idm_mse"]
    assert runs["good"]["idm"]["idm_mse"] == pytest.approx(gt_idm_mse, abs=1e-6)
    assert runs["good"]["idm_decodability_gap"] == pytest.approx(0.0, abs=1e-6)

    # The IDM decodability axis is sensitive to future quality: destroying the
    # future frames changes how well the action decodes, so the gap is non-trivial.
    assert abs(runs["bad"]["idm_decodability_gap"]) > 1e-6

    # Rankings cover both pixel quality and IDM action decodability.
    assert set(summary["rankings"].keys()) == {"by_idm_mse", "by_idm_decodability_gap", "by_future_mse"}
    assert summary["rankings"]["by_future_mse"] == ["good", "bad"]
    assert summary["best"]["by_future_mse"] == "good"
    assert summary["rank_by"] == "idm_mse"
    assert [run["label"] for run in summary["ranked"]] == summary["rankings"]["by_idm_mse"]

    # The summary is persisted and the per-cache pixel artifacts exist.
    written = json.loads((tmp_path / "ranking" / "ranking_summary.json").read_text())
    assert written["rankings"] == summary["rankings"]
    assert written["best"] == summary["best"]
    assert Path(runs["good"]["pixel"]["metrics_path"]).exists()


def test_rank_by_future_mse_orders_ranked_list(tmp_path) -> None:
    good_cache = build_dataset_future_cache(tmp_path / "good_cache", samples=6)
    bad_cache = build_dataset_future_cache(tmp_path / "bad_cache", samples=6)
    zero_out_cache_futures(bad_cache)
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    summary = rank_wan_lora_checkpoints(
        make_rank_args(
            idm_checkpoint,
            (bad_cache, good_cache),
            tmp_path / "ranking",
            labels=("bad", "good"),
            rank_by="future_mse",
        )
    )

    assert [run["label"] for run in summary["ranked"]] == ["good", "bad"]
    assert summary["best"]["by_future_mse"] == "good"


def test_idm_decodability_gap_sign_convention_and_orderings(tmp_path) -> None:
    good_cache = build_dataset_future_cache(tmp_path / "good_cache", samples=6)
    bad_cache = build_dataset_future_cache(tmp_path / "bad_cache", samples=6)
    zero_out_cache_futures(bad_cache)
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    summary = rank_wan_lora_checkpoints(
        make_rank_args(idm_checkpoint, (good_cache, bad_cache), tmp_path / "ranking", labels=("good", "bad"))
    )
    runs = {run["label"]: run for run in summary["runs"]}
    gt_idm_mse = summary["ground_truth_reference"]["idm_mse"]

    # Sign convention: the gap is generated-minus-ground-truth, so it is positive exactly when the
    # generated futures are *harder* to decode than the real ones. Pin it per run so a flipped
    # subtraction (which would invert "smaller is better" and rank the wrong checkpoints) is caught.
    for run in summary["runs"]:
        assert run["idm_decodability_gap"] == pytest.approx(run["idm"]["idm_mse"] - gt_idm_mse, abs=1e-9)
    # The exact ground-truth cache decodes actions identically to the reference: gap is exactly 0.
    assert runs["good"]["idm_decodability_gap"] == pytest.approx(0.0, abs=1e-6)
    # Destroying the futures shifts decodability, so the gap is non-trivial (its sign is exercised
    # against zero, not hard-coded: a near-future-blind IDM can land on either side of the GT).
    assert abs(runs["bad"]["idm_decodability_gap"]) > 1e-6

    # by_idm_mse sorts ascending on each run's idm_mse; by_idm_decodability_gap sorts ascending on
    # |gap|. Recompute both from the published per-run metrics so the orderings are pinned to the
    # right key and direction without hard-coding labels that depend on a tiny IDM's exact numbers.
    expected_by_idm_mse = [run["label"] for run in sorted(summary["runs"], key=lambda run: run["idm"]["idm_mse"])]
    expected_by_gap = [
        run["label"] for run in sorted(summary["runs"], key=lambda run: abs(run["idm_decodability_gap"]))
    ]
    assert summary["rankings"]["by_idm_mse"] == expected_by_idm_mse
    assert summary["rankings"]["by_idm_decodability_gap"] == expected_by_gap

    # The exact ground-truth cache (|gap| == 0, the global minimum) is always the most faithful, so
    # it ranks first on the decodability axis regardless of which side of zero the other caches land.
    # (The abs-vs-signed distinction itself is pinned by test_rank_key_idm_decodability_gap_ranks_by_magnitude.)
    assert summary["rankings"]["by_idm_decodability_gap"][0] == "good"
    assert summary["best"]["by_idm_decodability_gap"] == "good"


def test_rank_rejects_missing_idm_checkpoint(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=4, num_future_frames=4, action_horizon=8)

    with pytest.raises(FileNotFoundError, match="IDM checkpoint"):
        validate_ranking_inputs(make_rank_args(tmp_path / "missing.pt", (cache,), tmp_path / "ranking"))


def test_rank_rejects_missing_cache_dir(tmp_path) -> None:
    idm_checkpoint = tmp_path / "idm.pt"
    idm_checkpoint.write_text("stub\n")

    with pytest.raises(FileNotFoundError, match="cache directory not found"):
        validate_ranking_inputs(make_rank_args(idm_checkpoint, (tmp_path / "nope",), tmp_path / "ranking"))


def test_rank_rejects_missing_cache_config(tmp_path) -> None:
    idm_checkpoint = tmp_path / "idm.pt"
    idm_checkpoint.write_text("stub\n")
    cache = build_dataset_future_cache(tmp_path / "cache", samples=4)
    (cache / "config.json").unlink()

    with pytest.raises(FileNotFoundError, match="config"):
        validate_ranking_inputs(make_rank_args(idm_checkpoint, (cache,), tmp_path / "ranking"))


def test_rank_rejects_label_count_mismatch(tmp_path) -> None:
    idm_checkpoint = tmp_path / "idm.pt"
    idm_checkpoint.write_text("stub\n")
    cache = build_dataset_future_cache(tmp_path / "cache", samples=4)

    with pytest.raises(ValueError, match="labels"):
        validate_ranking_inputs(make_rank_args(idm_checkpoint, (cache,), tmp_path / "ranking", labels=("a", "b")))


def test_rank_rejects_cache_sample_count_mismatch(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=6)
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    with pytest.raises(ValueError, match="samples"):
        rank_wan_lora_checkpoints(
            make_rank_args(
                idm_checkpoint,
                (cache,),
                tmp_path / "ranking",
                max_samples=4,
                synthetic_samples=4,
            )
        )


def rewrite_manifest_dataset_indices(cache_dir: Path, new_indices) -> None:
    """Overwrite only the ``dataset_index`` column of a cache manifest, keeping length and tensors.

    This forges the exact failure modes the audit flagged: a same-length manifest whose rows
    duplicate, shuffle, or step outside the selected base dataset.
    """

    manifest_path = cache_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(rows) != len(new_indices):
        raise AssertionError(f"new_indices ({len(new_indices)}) must match manifest rows ({len(rows)})")
    for row, new_index in zip(rows, new_indices):
        row["dataset_index"] = new_index
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def mark_cache_as_wan_with_selected_indices(cache_dir: Path, selected_frame_indices: list[int]) -> None:
    manifest_path = cache_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["source"] = "wan_lora"
        row["future_frame_strategy"] = "first"
        row["selected_frame_indices"] = selected_frame_indices
        row["total_video_frames"] = max(selected_frame_indices) + 1
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    config_path = cache_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["future_source"] = "wan_lora"
    config["future_frame_selection"] = {
        "future_frame_strategy": "first",
        "selected_frame_indices": selected_frame_indices,
        "selected_frame_indices_by_dataset_index": {
            str(int(row["dataset_index"])): selected_frame_indices for row in rows
        },
        "total_video_frames": max(selected_frame_indices) + 1,
        "dataset_frame_delta": config["dataset_config"]["frame_delta"],
        "frame_delta": config["dataset_config"]["frame_delta"],
        "source_frame_offsets": expected_wan_source_frame_offsets(
            config["dataset_config"]["frame_delta"],
            config["dataset_config"]["num_future_frames"],
        ),
        "num_future_frames": config["dataset_config"]["num_future_frames"],
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")


# --- Temporal contract: --frame-delta must match the IDM training frame_delta ---


def test_load_idm_training_frame_delta_reads_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "idm.pt"
    torch.save({"train_config": {"dataset": {"frame_delta": 4}}}, checkpoint)

    assert load_idm_training_frame_delta(checkpoint) == 4


def test_load_idm_training_frame_delta_returns_none_without_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "idm.pt"
    torch.save({"model_config": {"image_size": 32}}, checkpoint)

    assert load_idm_training_frame_delta(checkpoint) is None


def test_enforce_frame_delta_contract_rejects_mismatch(tmp_path) -> None:
    checkpoint = tmp_path / "idm.pt"
    torch.save({"train_config": {"dataset": {"frame_delta": 4}}}, checkpoint)

    with pytest.raises(ValueError, match="disagrees with the IDM training"):
        enforce_idm_frame_delta_contract(checkpoint, requested_frame_delta=1)


def test_enforce_frame_delta_contract_accepts_match(tmp_path) -> None:
    checkpoint = tmp_path / "idm.pt"
    torch.save({"train_config": {"dataset": {"frame_delta": 4}}}, checkpoint)

    assert enforce_idm_frame_delta_contract(checkpoint, requested_frame_delta=4) == 4


def test_enforce_frame_delta_contract_allows_missing_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "idm.pt"
    torch.save({"model_config": {"image_size": 32}}, checkpoint)

    # No recorded frame_delta means nothing to enforce; warn instead of guessing a default.
    with pytest.warns(RuntimeWarning, match="does not record its training frame_delta"):
        assert enforce_idm_frame_delta_contract(checkpoint, requested_frame_delta=7) is None


def test_rank_rejects_frame_delta_mismatch_with_idm(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=6, frame_delta=1)
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6, frame_delta=1)

    # The IDM was trained for frame_delta=1; ranking with --frame-delta 2 must fail loudly.
    with pytest.raises(ValueError, match="disagrees with the IDM training"):
        rank_wan_lora_checkpoints(make_rank_args(idm_checkpoint, (cache,), tmp_path / "ranking", frame_delta=2))


# --- Temporal contract: Wan generated-video indices and source offsets are separate ---


def test_expected_selected_frame_indices_uses_generated_video_steps() -> None:
    assert expected_selected_frame_indices(frame_delta=1, num_future_frames=4) == [1, 2, 3, 4]
    assert expected_selected_frame_indices(frame_delta=4, num_future_frames=1) == [1]
    assert expected_selected_frame_indices(frame_delta=4, num_future_frames=2) == [1, 2]
    assert expected_selected_frame_indices(
        frame_delta=4,
        num_future_frames=4,
        strategy="source_offsets",
    ) == [4, 8, 12, 16]


def test_validate_cache_temporal_contract_accepts_aligned_wan_indices(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=2, frame_delta=1, num_future_frames=4)
    mark_cache_as_wan_with_selected_indices(cache, [1, 2, 3, 4])
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]

    validate_cache_temporal_contract(
        "aligned",
        cache,
        rows,
        requested_frame_delta=1,
        requested_num_future_frames=4,
    )


def test_validate_cache_temporal_contract_accepts_frame_delta_four_with_source_offsets(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=2, frame_delta=4, num_future_frames=1)
    mark_cache_as_wan_with_selected_indices(cache, [1])
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]

    validate_cache_temporal_contract(
        "aligned",
        cache,
        rows,
        requested_frame_delta=4,
        requested_num_future_frames=1,
    )


def test_validate_cache_temporal_contract_rejects_multiplied_generated_indices(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=2, frame_delta=4, num_future_frames=2)
    mark_cache_as_wan_with_selected_indices(cache, [4, 8])
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]

    with pytest.raises(ValueError, match=r"generated-video frame contract \[1, 2\]"):
        validate_cache_temporal_contract(
            "misaligned",
            cache,
            rows,
            requested_frame_delta=4,
            requested_num_future_frames=2,
        )


def test_validate_cache_temporal_contract_rejects_source_offset_mismatch(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=2, frame_delta=4, num_future_frames=1)
    mark_cache_as_wan_with_selected_indices(cache, [1])
    config_path = cache / "config.json"
    config = json.loads(config_path.read_text())
    config["future_frame_selection"]["source_frame_offsets"] = [1]
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    rows = [json.loads(line) for line in (cache / "manifest.jsonl").read_text().splitlines() if line.strip()]

    with pytest.raises(ValueError, match=r"source_frame_offsets \[1\].*\[4\]"):
        validate_cache_temporal_contract(
            "misaligned",
            cache,
            rows,
            requested_frame_delta=4,
            requested_num_future_frames=1,
        )


def test_rank_rejects_wan_cache_selected_frame_mismatch_with_idm_contract(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=6, frame_delta=4, num_future_frames=2)
    mark_cache_as_wan_with_selected_indices(cache, [4, 8])
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6, frame_delta=4, num_future_frames=2)

    with pytest.raises(ValueError, match="generated-video frame contract \\[1, 2\\]"):
        rank_wan_lora_checkpoints(
            make_rank_args(
                idm_checkpoint,
                (cache,),
                tmp_path / "ranking",
                frame_delta=4,
            )
        )


# --- Cache sample identity: manifest dataset_index must be exactly range(len(base_dataset)) ---


def test_validate_cache_sample_identity_accepts_generated_sequence() -> None:
    cached_dataset = SimpleNamespace(rows=[{"dataset_index": index} for index in range(6)])

    # A freshly generated cache addresses every base sample once, in order: this must pass.
    validate_cache_sample_identity("good", cached_dataset, expected_num_samples=6)


def test_validate_cache_sample_identity_rejects_duplicate() -> None:
    cached_dataset = SimpleNamespace(rows=[{"dataset_index": i} for i in [0, 1, 2, 3, 4, 4]])

    with pytest.raises(ValueError, match="repeats dataset_index"):
        validate_cache_sample_identity("dupe", cached_dataset, expected_num_samples=6)


def test_validate_cache_sample_identity_rejects_shuffled() -> None:
    cached_dataset = SimpleNamespace(rows=[{"dataset_index": i} for i in [5, 4, 3, 2, 1, 0]])

    with pytest.raises(ValueError, match="sequence does not match"):
        validate_cache_sample_identity("shuffled", cached_dataset, expected_num_samples=6)


def test_validate_cache_sample_identity_rejects_out_of_range() -> None:
    cached_dataset = SimpleNamespace(rows=[{"dataset_index": i} for i in [0, 1, 2, 3, 4, 6]])

    with pytest.raises(ValueError, match="outside the base dataset range"):
        validate_cache_sample_identity("oob", cached_dataset, expected_num_samples=6)


def test_rank_rejects_duplicate_dataset_index_manifest(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=6)
    # Same length as the base dataset, but one base sample is counted twice and another dropped.
    rewrite_manifest_dataset_indices(cache, [0, 1, 2, 3, 4, 4])
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    with pytest.raises(ValueError, match="repeats dataset_index"):
        rank_wan_lora_checkpoints(make_rank_args(idm_checkpoint, (cache,), tmp_path / "ranking"))


def test_rank_rejects_shuffled_dataset_index_manifest(tmp_path) -> None:
    cache = build_dataset_future_cache(tmp_path / "cache", samples=6)
    # Same length and the same index set, but reordered so futures pair with the wrong base sample.
    rewrite_manifest_dataset_indices(cache, [5, 4, 3, 2, 1, 0])
    idm_checkpoint = train_tiny_idm(tmp_path / "idm", samples=6)

    with pytest.raises(ValueError, match="sequence does not match"):
        rank_wan_lora_checkpoints(make_rank_args(idm_checkpoint, (cache,), tmp_path / "ranking"))
