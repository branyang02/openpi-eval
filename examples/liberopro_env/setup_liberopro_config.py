"""Write LIBERO-Pro's default config.yaml for this checkout."""

from __future__ import annotations

import pathlib


def build_config_text() -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    libero_root = repo_root / "third_party" / "liberopro" / "libero" / "libero"
    datasets_root = repo_root / "third_party" / "liberopro" / "libero" / "datasets"
    return "\n".join(
        [
            "benchmark_root: {}".format(libero_root),
            "bddl_files: {}".format(libero_root / "bddl_files"),
            "init_states: {}".format(libero_root / "init_files"),
            "datasets: {}".format(datasets_root),
            "assets: {}".format(libero_root / "assets"),
            "",
        ]
    )


def setup_liberopro_config() -> pathlib.Path:
    config_dir = pathlib.Path.home() / ".liberopro"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "config.yaml"
    config_path.write_text(build_config_text())
    return config_path


def main() -> None:
    config_path = setup_liberopro_config()
    print(f"Wrote {config_path}")


if __name__ == "__main__":
    main()
