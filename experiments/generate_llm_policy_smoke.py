from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_policy.llm_code_generator import OpenAICompatiblePolicyGenerator


TASK = "Right-angle left turn. Approach a fixed marker, turn left, hold the arc, then return center after the heading target."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "runs" / "llm_policy_smoke"))
    parser.add_argument("--feedback-event", default="turn_too_early")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    generator = OpenAICompatiblePolicyGenerator()
    result = generator.generate(
        TASK,
        feedback_event=args.feedback_event,
        previous_failure="initial attempt turned too early and cut inside the corner",
    )
    (out / "policy.py").write_text(result.code)
    summary = {
        "model": result.model,
        "base_url": result.base_url,
        "feedback_event": result.feedback_event,
        "policy_path": str(out / "policy.py"),
        "validated": True,
        "validation_errors": list(result.validation_errors),
        "repair_attempts": result.repair_attempts,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
