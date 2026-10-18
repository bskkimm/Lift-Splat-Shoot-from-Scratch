import subprocess,sys
def test_train_cli_exposes_checkpoint_name():
    r=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "--checkpoint-name" in r.stdout
