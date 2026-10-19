import subprocess,sys
def test_train_dry_run_reports_status():
    r=subprocess.run([sys.executable,"train.py","--dry-run"],capture_output=True,text=True); assert "dry_run=True" in r.stdout
