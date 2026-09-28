import subprocess,sys
def test_eval_cli_exposes_seed():
    r=subprocess.run([sys.executable,"eval.py","--help"],capture_output=True,text=True); assert "--seed" in r.stdout
