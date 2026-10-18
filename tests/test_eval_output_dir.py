import subprocess,sys
def test_eval_cli_exposes_output_dir():
    r=subprocess.run([sys.executable,"eval.py","--help"],capture_output=True,text=True); assert "--output-dir" in r.stdout
