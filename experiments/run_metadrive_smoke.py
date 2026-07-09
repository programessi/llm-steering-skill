from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.metadrive_adapter import MetaDriveAdapter
from envs.simulator_adapter import TaskConfig
from llm_policy.code_generator import PolicyCodeGenerator
from llm_policy.policy_runtime import RestrictedPolicyRuntime
from skills.oracle_steering_skill import OracleSteeringSkill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "metadrive_smoke"))
    parser.add_argument("--horizon", type=int, default=120)
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env = MetaDriveAdapter()
    try:
        env.reset(TaskConfig(name="metadrive_lane_keep", horizon_steps=args.horizon))
        skill = OracleSteeringSkill(env)
        code = PolicyCodeGenerator(use_codex=False).generate("MetaDrive lane keeping", mode="llm_feedback")
        runtime = RestrictedPolicyRuntime(env=env, steering_skill=skill, use_oracle_state=True)
        stats = runtime.run(code)
        result = {
            **env.metrics(),
            "state_source": "metadrive_oracle",
            "policy_mode": "llm_feedback",
            "valid_generated_code": stats.valid_code,
            "primitive_call_count": stats.primitive_call_count,
            "events": stats.events,
        }
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(result, indent=2))
        (out / "policy.py").write_text(code)
        with (out / "trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(env.trace[0].keys()))
            writer.writeheader()
            writer.writerows(env.trace)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
