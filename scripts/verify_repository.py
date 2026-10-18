"""Check payload hashes, extraction pins and data exclusions."""
from pathlib import Path
import argparse,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,default=ROOT/'MANIFEST.json');a=p.parse_args();m=json.loads(a.manifest.read_text());errors=[]
for row in m['files']:
 f=ROOT/row['path']
 if not f.exists() or f.stat().st_size!=row['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:errors.append(row['path'])
def included(f):
 rel=f.relative_to(ROOT)
 if any(x in rel.parts for x in ['.git','build','__pycache__','.venv','.pytest_cache','external_data']):return False
 if rel.parts[0]=='engine' and any(x in rel.parts for x in ['data','results','logs']):return False
 return f.is_file() and f.suffix not in ['.pyc','.pyo'] and rel.as_posix()!='MANIFEST.json'
actual={f.relative_to(ROOT).as_posix() for f in ROOT.rglob('*') if included(f)}
expected={row['path'] for row in m['files']}
if actual!=expected:errors.append({'coverage_missing':sorted(expected-actual),'coverage_extra':sorted(actual-expected)})
extraction=json.loads((ROOT/'provenance/extraction_manifest.json').read_text())
variants={v['path']:v for v in extraction.get('public_variants',[])}
for row in extraction['files']:
 if row['public_path'].startswith('engine/'):
  f=ROOT/row['public_path']
  variant=variants.get(row['public_path'])
  if variant and variant['original_sha256']!=row['sha256']:errors.append('variant original pin: '+row['public_path'])
  expected_pin=variant['public_sha256'] if variant else row['sha256']
  if hashlib.sha256(f.read_bytes()).hexdigest()!=expected_pin:errors.append('preserved source/public variant: '+row['public_path'])
if 'magnitudes' in json.loads((ROOT/'paper/data/catalogue.json').read_text()):errors.append('event-level magnitude array')
if list((ROOT/'paper').rglob('manuscript_v2.pdf')):errors.append('earlier manuscript')
print(json.dumps({'checked_files':len(m['files']),'errors':errors},indent=2));raise SystemExit(bool(errors))
