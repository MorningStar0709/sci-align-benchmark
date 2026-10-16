import re


def iter_markdown_files(input_path):
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*.md") if not path.stem.endswith("_original"))


def strip_code_fences(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)
