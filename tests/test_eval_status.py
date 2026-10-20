import subprocess,sys
def test_eval_help_remains_available():
    r=subprocess.run([sys.executable,"eval.py","--help"],capture_output=True,text=True); assert r.returncode==0
