import subprocess,sys
def test_eval_help_has_output_option():
    r=subprocess.run([sys.executable,"eval.py","--help"],capture_output=True,text=True); assert "--output" in r.stdout
