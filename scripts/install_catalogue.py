"""Verify external historical CSV and install into an ignored input path."""
from pathlib import Path
import argparse,hashlib,json,shutil,csv
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('csv',type=Path);p.add_argument('--check-only',action='store_true');a=p.parse_args()
data=json.loads((ROOT/'engine/iteration_27/configs/observed_frozen.json').read_text())['data'];actual=hashlib.sha256(a.csv.read_bytes()).hexdigest()
if actual!=data['sha256']:raise SystemExit('Snapshot hash differs. A live download is a new snapshot, not the historical input. No file installed.')
with a.csv.open(encoding='utf-8-sig',newline='') as f:count=sum(1 for _ in csv.DictReader(f))
if count!=data['n']:raise SystemExit('Selected count differs')
if not a.check_only:
 dest=ROOT/'engine'/data['path'];dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():raise SystemExit('Refusing to overwrite existing input')
 shutil.copy2(a.csv,dest);print('Installed ignored local input:',dest)
print(json.dumps({'sha256':actual,'events':count,'check_only':a.check_only}))
