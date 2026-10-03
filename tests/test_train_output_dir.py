import subprocess,sys
def test_train_cli_exposes_output_dir():
    r=subprocess.run([sys.executable,"train.py","--help"],capture_output=True,text=True); assert "--output-dir" in r.stdout
