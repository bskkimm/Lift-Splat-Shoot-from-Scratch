import subprocess,sys
def test_train_cli_exposes_seed():
    result=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "--seed" in result.stdout
