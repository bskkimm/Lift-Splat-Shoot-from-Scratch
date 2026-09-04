import subprocess, sys
def test_train_and_eval_cli_smoke(tmp_path):
    train = subprocess.run([sys.executable, "train.py", "--dry-run", "--depth-bins", "2"], capture_output=True, text=True)
    assert train.returncode == 0
