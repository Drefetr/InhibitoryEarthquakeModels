"""Explicit frozen observed fitting; exact external input required."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];engine=ROOT/'engine'
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('build/observed.json'));p.add_argument('--check-inputs-only',action='store_true');a=p.parse_args()
protocol=engine/'iteration_27/configs/observed_frozen.json';spec=json.loads(protocol.read_text());errors=[]
for rel,expected in spec['source_hashes'].items():
 f=engine/rel
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:errors.append(rel)
data=engine/spec['data']['path']
if not data.exists() or hashlib.sha256(data.read_bytes()).hexdigest()!=spec['data']['sha256']:errors.append('external catalogue: see docs/data.md')
if errors:raise SystemExit('Required inputs missing or changed:\n'+'\n'.join(errors))
if a.check_inputs_only:print('Frozen sources and external catalogue verified');raise SystemExit(0)
a.output.parent.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,str(engine/'iteration_27/code/run_adequacy.py'),'--protocol',str(protocol),'--protocol-sha256',hashlib.sha256(protocol.read_bytes()).hexdigest(),'--output',str(a.output.resolve())]
raise SystemExit(subprocess.call(cmd,cwd=ROOT))
