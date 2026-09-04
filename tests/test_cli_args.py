import subprocess, sys
def test_train_cli_exposes_model_arguments():
    result=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "depth-bins" in result.stdout
