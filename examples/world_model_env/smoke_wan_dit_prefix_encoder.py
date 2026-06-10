from __future__ import annotations

import tyro

from world_model.wan_dit_prefix_encoder import WanDiTRandomSmokeArgs, smoke_main

if __name__ == "__main__":
    smoke_main(tyro.cli(WanDiTRandomSmokeArgs))
