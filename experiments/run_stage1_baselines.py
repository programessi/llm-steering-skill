from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.kinematic_simulator import make_stage1_tasks
from experiments.run_stage1_demo import run_one


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "stage1_baselines"))
    parser.add_argument("--video-task", default="curve_following")
    args = parser.parse_args()

    out_root = Path(args.out)
    summaries = []
    for task in make_stage1_tasks():
        for state_source in ("oracle", "perceived"):
            for policy_mode in ("rule", "llm_no_feedback", "llm_feedback"):
                out_dir = out_root / f"{task.name}_{state_source}_{policy_mode}"
                summaries.append(
                    run_one(
                        task.name,
                        state_source,
                        out_dir,
                        make_video=(
                            task.name == args.video_task
                            and state_source == "perceived"
                            and policy_mode == "llm_feedback"
                        ),
                        policy_mode=policy_mode,
                    )
                )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
