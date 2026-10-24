import subprocess,sys
def test_train_cli_exposes_scheduler():
    r=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "--scheduler" in r.stdout
