"""Read-only source/PDF production checks and rendering; no scientific execution."""
from pathlib import Path
import argparse,json,hashlib,re
from collections import Counter
import pymupdf as fitz
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--pdf',type=Path,required=True);p.add_argument('--log',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);pages=a.output/'pages';pages.mkdir()
tex='\n'.join(x.read_text(encoding='utf-8') for x in (a.source_root/'source').rglob('*.tex'))
labels=Counter(re.findall(r'\\label\{([^}]+)\}',tex));refs=re.findall(r'\\(?:[Cc]ref|[Ee]qref|ref)\{([^}]+)\}',tex)
cites={k.strip() for r in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',tex) for k in r.split(',')}
bib=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(a.source_root/'source/references.bib').read_text(encoding='utf-8')))
log=a.log.read_text(encoding='utf-8',errors='replace');doc=fitz.open(a.pdf);tiles=[];texts=[];edges=[]
for i,page in enumerate(doc):
 pix=page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False);target=pages/f'page_{i+1:03d}.png';pix.save(target)
 im=Image.open(target);im.thumbnail((300,425));tile=Image.new('RGB',(320,455),'#eeeeee');tile.paste(im,((320-im.width)//2,0));ImageDraw.Draw(tile).text((12,435),f'Page {i+1}',fill='black');tiles.append(tile)
 texts.append(page.get_text())
 for b in page.get_text('blocks'):
  if b[0]<18 or b[2]>page.rect.width-18:edges.append({'page':i+1,'bbox':b[:4],'text':b[4][:100]})
for start in range(0,len(tiles),12):
 sheet=Image.new('RGB',(1280,1365),'white')
 for j,t in enumerate(tiles[start:start+12]):sheet.paste(t,((j%4)*320,(j//4)*455))
 sheet.save(a.output/f'contact_{start//12+1:02d}.png')
(a.output/'extracted_text.txt').write_text('\n\f\n'.join(texts),encoding='utf-8')
figs=re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex)
c={'pdf_sha256':hashlib.sha256(a.pdf.read_bytes()).hexdigest(),'pages':len(doc),'duplicate_labels':[k for k,v in labels.items() if v>1],'undefined_cross_references':sorted({k.strip() for r in refs for k in r.split(',')}-set(labels)),'missing_bibliography_keys':sorted(cites-bib),'missing_figures':[f for f in figs if not (a.source_root/'figures'/f).exists()],'overfull_boxes':len(re.findall(r'Overfull \\hbox',log)),'undefined_warnings':len(re.findall('undefined',log,re.I)),'missing_glyph_warnings':len(re.findall(r'Missing character|missing glyph',log,re.I)),'underfull_boxes':len(re.findall(r'Underfull \\hbox',log)),'replacement_characters':sum(t.count('\ufffd') for t in texts),'edge_text_blocks':edges,'render_scale':1.7,'scope':'production checks; page images require visual review'}
c['passed']=not any(c[k] for k in ['duplicate_labels','undefined_cross_references','missing_bibliography_keys','missing_figures','overfull_boxes','undefined_warnings','missing_glyph_warnings','replacement_characters','edge_text_blocks'])
(a.output/'production_checks.json').write_text(json.dumps(c,indent=2)+'\n',encoding='utf-8');print(json.dumps(c,indent=2))
if not c['passed']:raise SystemExit(1)
