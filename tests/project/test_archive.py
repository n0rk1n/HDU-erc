import ast
from pathlib import Path
import subprocess


BASELINE = "86220ac52df1641fee051bae8e4e04fd09ff6e40"
ARCHIVE = Path("archive/emotion-aware-chatbot-v1")


def test_every_baseline_file_exists_under_archive():
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE], text=True
    )
    baseline_files = output.splitlines()
    assert len(baseline_files) == 166
    missing = [path for path in baseline_files if not (ARCHIVE / path).is_file()]
    assert missing == []


def test_new_runtime_does_not_import_archive() -> None:
    """Catches executable Python imports coupling the new runtime to frozen v1 code."""
    sources = list(Path("chatbot").rglob("*.py"))
    assert sources
    offenders: list[tuple[Path, int, str]] = []
    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
            for module in imported:
                if module == "archive" or module.startswith("archive."):
                    offenders.append((path, node.lineno, module))
    assert offenders == []
