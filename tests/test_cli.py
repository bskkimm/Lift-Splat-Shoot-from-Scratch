import subprocess, sys
def test_train_cli_help():
    result = subprocess.run([sys.executable, "train.py", "--help"], capture_output=True, text=True); assert result.returncode == 0; assert "epochs" in result.stdout
