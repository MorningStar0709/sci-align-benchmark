"""Control science evaluation domain."""
from mineru_eval.domains import DomainConfig, DomainRegistry

DomainRegistry.register(DomainConfig(
    name="control",
    dimensions=["A", "B", "C", "D"],
    scorer_module="mineru_eval.control.scorers",
    prompt_module="mineru_eval.control.prompts",
    target_system_prompt="你是一个控制科学领域的专家。请回答以下控制科学问题。",
    default_input="benchmark/dataset/sample_questions.json",
    default_output="benchmark/dataset/sample_report.json",
))
