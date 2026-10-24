import subprocess,sys
def test_train_cli_exposes_optimizer_args():
    r=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "--lr" in r.stdout and "--weight-decay" in r.stdout
