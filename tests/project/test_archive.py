from pathlib import Path
import subprocess


BASELINE = "86220ac52df1641fee051bae8e4e04fd09ff6e40"
ARCHIVE = Path("archive/emotion-aware-chatbot-v1")


def test_every_baseline_file_exists_under_archive():
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE], text=True
    )
    missing = [path for path in output.splitlines() if not (ARCHIVE / path).is_file()]
    assert missing == []
