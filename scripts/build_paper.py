"""Build paper and supplement; no scientific code is executed."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,platform
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--engine',default='tectonic');p.add_argument('--output',type=Path,default=Path('build/paper'));a=p.parse_args()
engine=shutil.which(a.engine) or str(Path(a.engine).resolve());version=subprocess.check_output([engine,'--version'],text=True).strip()
if version!='Tectonic 0.17.0':raise SystemExit('Expected Tectonic 0.17.0; got '+version)
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
record={'engine':str(engine),'engine_version':version,'engine_sha256':hashlib.sha256(Path(engine).read_bytes()).hexdigest(),'platform':platform.platform(),'historical_cache':'unsnapshotted; no historical exact replay claim','commands':[]}
for source,name in [(ROOT/'paper/source/main.tex','manuscript.pdf'),(ROOT/'paper/supplements/explanatory_supplement.tex','explanatory_supplement.pdf')]:
 target=out/source.stem;target.mkdir();cmd=[engine,source.name,'--outdir',str(target),'--keep-logs','--keep-intermediates','--makefile-rules',str(target/'dependencies.mk')]
 proc=subprocess.run(cmd,cwd=source.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (target/'console.log').write_text(proc.stdout,encoding='utf-8');record['commands'].append({'command':cmd,'cwd':str(source.parent),'exit_code':proc.returncode})
 (out/'build_record.json').write_text(json.dumps(record,indent=2));print(proc.stdout)
 if proc.returncode:raise SystemExit(proc.returncode)
 shutil.copy2(target/(source.stem+'.pdf'),out/name);record.setdefault('pdfs',{})[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
(out/'build_record.json').write_text(json.dumps(record,indent=2));print('Built PDFs in',out)
