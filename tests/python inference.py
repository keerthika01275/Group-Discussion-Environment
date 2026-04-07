import subprocess
import sys


def test_inference_runs():
    result = subprocess.run(
        [sys.executable, "inference.py"],
        capture_output=True,
        text=True,
        check=True
    )

    output = result.stdout
    assert "[START]" in output
    assert "[STEP]" in output
    assert "[END]" in output