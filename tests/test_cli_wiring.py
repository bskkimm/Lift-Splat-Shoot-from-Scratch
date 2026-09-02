import subprocess, sys
def test_train_dry_run_builds_model():
    result=subprocess.run([sys.executable,"train.py","--dry-run"],capture_output=True,text=True); assert result.returncode == 0 and "parameters=" in result.stdout
