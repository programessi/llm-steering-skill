from __future__ import annotations

from dataclasses import dataclass

from llm_policy.openai_compatible_client import (
    LLMClientConfig,
    OpenAICompatibleClient,
    extract_python_code,
)
from llm_policy.prompt_builder import PolicyPromptInput, build_policy_messages
from llm_policy.safety_checker import validate_generated_policy_code


@dataclass(frozen=True)
class GeneratedPolicyCode:
    code: str
    model: str
    base_url: str
    feedback_event: str
    validation_errors: tuple[str, ...] = ()
    repair_attempts: int = 0


class OpenAICompatiblePolicyGenerator:
    def __init__(self, config: LLMClientConfig | None = None, max_repair_attempts: int = 1):
        self.config = config or LLMClientConfig.from_env()
        self.client = OpenAICompatibleClient(self.config)
        self.max_repair_attempts = max(0, int(max_repair_attempts))

    def generate(self, task_description: str, feedback_event: str = "none", previous_failure: str = "none") -> GeneratedPolicyCode:
        validation_error = "none"
        last_errors: tuple[str, ...] = ()
        for repair_attempt in range(self.max_repair_attempts + 1):
            messages = build_policy_messages(
                PolicyPromptInput(
                    task_description=task_description,
                    feedback_event=feedback_event,
                    previous_failure=previous_failure,
                    validation_error=validation_error,
                )
            )
            raw = self.client.chat(messages)
            code = extract_python_code(raw)
            validation = validate_generated_policy_code(code)
            if validation.ok:
                return GeneratedPolicyCode(
                    code=code,
                    model=self.config.model,
                    base_url=self.config.base_url,
                    feedback_event=feedback_event,
                    validation_errors=(),
                    repair_attempts=repair_attempt,
                )
            last_errors = validation.errors
            validation_error = "; ".join(validation.errors)
        raise RuntimeError(f"Generated policy failed validation: {'; '.join(last_errors)}")
