"""Medical cross-domain evaluation domain."""
from mineru_eval.domains import DomainConfig, DomainRegistry

DomainRegistry.register(DomainConfig(
    name="medical",
    dimensions=["identify", "model", "transfer"],
    default_difficulty_levels=["L1", "L2", "L3", "L4"],
    scorer_module="mineru_eval.medical.scorers",
    prompt_module="mineru_eval.medical.prompts",
    target_system_prompt="你是一个医学与控制科学跨领域专家。请回答以下医学场景中的控制理论问题。",
    default_input="benchmark/dataset/medical_benchmark.json",
    default_output="benchmark/dataset/medical_report.json",
))
