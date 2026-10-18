"""Render saved evidence and explicitly algebraic diagrams; never fit or simulate."""
from pathlib import Path
import csv, json, hashlib, math
import sys
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('build/figures'))
args=parser.parse_args()
if args.output.exists():raise FileExistsError('Use a new figure output directory')
DRAFT=Path(__file__).resolve().parents[1]/'paper'
ROOT=DRAFT.parents[1]
FIG=args.output.resolve(); DATA=DRAFT/'data'; OUTDATA=FIG/'data'; OUTDATA.mkdir(parents=True)
FIG.mkdir(exist_ok=True)
PORTABLE=True
OLD_MANIFEST=json.loads((DRAFT/'figures/figure_manifest.json').read_text()) if (DRAFT/'figures/figure_manifest.json').exists() else []
OLD_SOURCES={s['path']:s for f in OLD_MANIFEST for s in f['sources']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':10,'axes.titlesize':11,'savefig.bbox':'tight'})
BLUE='#0072B2'; ORANGE='#D55E00'; GRAY='#777777'
manifest=[]
def source(path):
    p=ROOT/path
    if PORTABLE: return {**OLD_SOURCES[path],'verification':'Inherited from shipped manifest; archival bytes not reread in portable mode.'}
    return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'verification':'SHA256 computed from local archival bytes during figure generation.'}
def save(fig,name,meaning,sources,derived=None):
    fig.savefig(FIG/f'{name}.pdf',metadata={'Title':name,'Creator':'Saved-evidence manuscript plotting','CreationDate':None,'ModDate':None}); fig.savefig(FIG/f'{name}.png',dpi=180); plt.close(fig)
    if derived is not None: (OUTDATA/f'{name}.json').write_text(json.dumps(derived,indent=2)+'\n',encoding='utf8')
    manifest.append({'figure':f'{name}.pdf','label':f'fig:{name}','meaning':meaning,'sources':[source(x) for x in sources],'derived_data':f'../data/{name}.json' if derived is not None else None})
def readcsv(p): return list(csv.DictReader((ROOT/p).open(encoding='utf-8-sig',newline='')))
def box(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.018',edgecolor=color,facecolor='#F5F8FA',linewidth=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.65)

# 1: conceptual map, no artificial history.
fig,ax=plt.subplots(figsize=(7.1,4.6)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
ax.text(.25,.97,'Physical loading model',ha='center',weight='bold'); ax.text(.76,.97,'Independent-background\ndiagnostic',ha='center',weight='bold')
box(ax,.02,.66,.43,.19,'Loading and event decrements\n'+r'$X(t)=X_0\Phi_s(t)+\rho\int_0^t\Phi_s(u)\,du-G_s(t)$')
box(ax,.53,.66,.43,.19,'Background and event memory\n'+r'$B(t),\quad G_s(t)=\sum_{t_i<t}g(Q_i)\Phi_s(t-t_i)$')
box(ax,.02,.32,.43,.19,'Physical intensity\n'+r'$\lambda_{\rm phys}=\Psi(X(t^-))+H_{\rm phys}(t)$')
box(ax,.53,.32,.43,.19,'Diagnostic intensity\n'+r'$\lambda_\eta=B(t)e^{-\eta G_s(t)}+nH_e(t)$')
for x in [.235,.745]: ax.annotate('',(x,.52),(x,.65),arrowprops={'arrowstyle':'->','lw':1.4})
ax.text(.5,.14,'Shared-memory restriction: '+r'$h_e=-\Phi_s^{\prime}$',ha='center',fontsize=12)
ax.text(.5,.035,'The background B is independent of the physical loading construction.',ha='center',color='#444444')
save(fig,'model_architecture','Conceptual distinction between physical loading and independent-background diagnostic intensity; arrows are model definitions, not evidence.',[],{'type':'conceptual','normalization':'E kappa=1 for diagnostic; physical productivity absorbs n'})

# 2: theorem scope, deliberately no implication from finite to infinite stability.
fig,ax=plt.subplots(figsize=(7.1,4.2)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
box(ax,.03,.64,.43,.25,'Finite-mode physical model\nFirst-moment drift / regeneration\nStated finite-mode assumptions')
box(ax,.54,.64,.43,.25,'Ideal infinite-memory model\nSeparate stability question\nNot settled by finite-mode proof',GRAY)
box(ax,.03,.15,.43,.28,'Fixed-history diagnostic geometry\nConditional scores / projections\nSelected numerical implementation')
box(ax,.54,.15,.43,.28,'Observed catalogue comparisons\nPredictive scores / residuals\nMechanism remains unidentified',ORANGE)
ax.text(.5,.51,'Distinct mathematical and evidential scopes',ha='center',fontsize=12,weight='bold')
save(fig,'theory_scope','Scope map separating finite-mode stability, ideal infinite-memory questions, conditional inference and observed predictive evidence.',[],{'type':'conceptual scope diagram','arrows':'none; no unproved implications'})

# 3: five procedures and each own matched reference bank.
p21='iteration_21/results/analysis/2026-10-17T004327174461_0000/summary.json'
p26='iteration_26/results/analysis/2026-10-17T125009077299_0000/summary.json'
if PORTABLE:
    portable=json.loads((DATA/'detection.json').read_text())
    j21={'methods':{'oracle':portable['Fixed-eta oracle'],'known_known':portable['Eta fit; background supplied'],'background_estimated':portable['Level and trend fit']}}
    j26={'methods':{'level_only':portable['Level fit; trend supplied']}}
else:
    j21=json.loads((ROOT/p21).read_text()); j26=json.loads((ROOT/p26).read_text())
p19='iteration_19/results/analysis/2026-10-16T144344808758_0000/summary.json'
if PORTABLE: fitted=portable['Historical fuller fit']
else:
    j19=json.loads((ROOT/p19).read_text()); fitted=dict(j19['methods']['fitted']); fitted['alternative_frequency']=fitted.pop('alternative_power')
methods=[('Fixed-eta oracle',j21['methods']['oracle']),('Eta fit; background supplied',j21['methods']['known_known']),('Level fit; trend supplied',j26['methods']['level_only']),('Level and trend fit',j21['methods']['background_estimated']),('Historical fuller fit',fitted)]
fig,axes=plt.subplots(1,2,figsize=(7.25,3.8),sharey=True,gridspec_kw={'width_ratios':[1.7,1]})
for ax,key,title,xlim in zip(axes,['alternative_frequency','heldout_null_size'],['Alternative detection\n150 queries','Held-out null rejection\n1,000 queries'],[(-3,108),(0,8.5)]):
    for i,(label,m) in enumerate(methods):
        r=m[key]; v=r['conditional_frequency']*100; lo,hi=np.array(r['exact_binomial_95_interval'])*100
        ax.errorbar(v,4-i,xerr=[[v-lo],[hi-v]],fmt='o',color=BLUE if key=='alternative_frequency' else ORANGE,capsize=3,markersize=5)
        ax.text(xlim[1]-.4,4-i+.19,f"{r['rejections']}/{r['requested']}",ha='right',fontsize=9)
    ax.set(xlim=xlim,ylim=(-.55,4.65),xlabel='Percent',title=title); ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(range(5),[x[0] for x in methods[::-1]]); axes[1].axvline(5,color=GRAY,ls=':',lw=1)
fig.tight_layout(w_pad=1.5)
save(fig,'detection','Nominal conditional Clopper-Pearson intervals for five finite procedures, each with its own 999-reference bank on the same query identities. Intervals exclude bank and selection uncertainty.',[p19,p21,p26],{name:m for name,m in methods})

# 4: paired 2x2 count display.
fig,ax=plt.subplots(figsize=(5.4,3.0)); vals=np.array([[62,61],[6,21]])
ax.imshow(vals,cmap='Blues',vmin=0,vmax=70,aspect='auto')
for i in range(2):
 for k in range(2): ax.text(k,i,str(vals[i,k]),ha='center',va='center',fontsize=27,color='white' if vals[i,k]>40 else '#222222')
ax.set_xticks([0,1],['Level-only: reject','Level-only: retain']); ax.set_yticks([0,1],['Level + trend:\nreject','Level + trend:\nretain'])
# reorder columns to align rows with counts both, BG-only vs level-only, neither
vals=np.array([[62,6],[61,21]]); ax.images[0].set_data(vals)
for txt in list(ax.texts): txt.remove()
for i in range(2):
 for k in range(2): ax.text(k,i,str(vals[i,k]),ha='center',va='center',fontsize=27,color='white' if vals[i,k]>40 else '#222222')
ax.set_title('Same 150 alternative histories',pad=14); ax.tick_params(length=0,pad=10)
save(fig,'paired','Paired level-only versus background-level-and-trend decisions. Both 62, level-only 61, level-and-trend-only 6, neither 21. No pathwise dominance.',[p26],{'both':62,'level_only':61,'level_and_trend_only':6,'neither':21,'total':150})

# 5: observed out-of-sample comparisons from saved scores.
pg='iteration_28/results/numerical_audit/global_decomposition.csv'
folds=['fold_50','fold_65','fold_80']
if PORTABLE: pred=json.loads((DATA/'prediction.json').read_text())['contrasts']
else:
    rows=[r for r in readcsv(pg) if r['order']=='32']
    pred={c:[float(next(r for r in rows if r['fold']==f and r['comparison']==c)['full_marked_logscore_difference']) for f in folds] for c in ['shared_minus_ml','omori_minus_ml']}
fig,ax=plt.subplots(figsize=(7.1,3.7)); xx=np.arange(3)
for offset,c,label,color in [(-.17,'shared_minus_ml','Shared ML minus ML',BLUE),(.17,'omori_minus_ml','Omori minus ML',ORANGE)]:
 bars=ax.bar(xx+offset,pred[c],width=.32,label=label,color=color)
 for bar,v in zip(bars,pred[c]): ax.annotate(f'{v:+.2f}',(bar.get_x()+bar.get_width()/2,v),xytext=(0,4 if v>=0 else -12),textcoords='offset points',ha='center',fontsize=9)
ax.axhline(0,color='#333333',lw=.8); ax.set_xticks(xx,['1993-2003','2003-2013','2013-2026']); ax.set(ylabel='Marked test logscore difference',ylim=(-19.5,10)); ax.legend(frameon=False,ncol=2,loc='upper center')
ax.text(.65,.12,'First period: 93.7% of the\naggregate shared-model loss',transform=ax.transAxes,fontsize=9,ha='center')
save(fig,'prediction','Retrospective observed test-score differences in three disjoint periods. No independent-fold uncertainty or causal interpretation is supplied.',[pg],{'folds':folds,'contrasts':pred,'first_period_fraction_shared_loss':pred['shared_minus_ml'][0]/sum(pred['shared_minus_ml'])})

# 6: actual temporal components; complete quiet exposure included.
pt='iteration_28/results/numerical_audit/temporal_comparisons.csv'
tr=json.loads((DATA/'temporal.json').read_text()) if PORTABLE else [r for r in readcsv(pt) if r['order']=='32' and r['comparison']=='shared_minus_ml']
fig,axes=plt.subplots(1,3,figsize=(7.2,3.45),sharey=True)
for ax,f,title in zip(axes,folds,['1993-2003','2003-2013','2013-2026']):
 rr=sorted([r for r in tr if r['fold']==f],key=lambda r:int(r['bin'])); x=np.arange(1,5)
 ax.bar(x-.16,[float(r['event_logratio']) for r in rr],width=.3,color=BLUE,label='Event term E')
 ax.bar(x+.16,[float(r['compensator_contribution']) for r in rr],width=.3,color=ORANGE,label='Compensator C')
 ax.plot(x,[float(r['full_marked_logscore_difference']) for r in rr],'o-',color='#222222',ms=4,label='Full score E + C')
 ax.axhline(0,color=GRAY,lw=.7); ax.set(title=title,xlabel='Equal-duration test bin',xticks=x)
axes[0].set_ylabel('Shared ML minus ML logscore term'); axes[1].legend(loc='upper center',bbox_to_anchor=(.5,1.32),ncol=3,frameon=False,fontsize=9)
fig.tight_layout(w_pad=1)
save(fig,'temporal','Saved full-score decomposition by four equal-duration bins per period, using updating full history and retained terminal exposure.',[pt],[{k:r[k] for k in ['fold','bin','event_logratio','compensator_contribution','full_marked_logscore_difference']} for r in tr])

# 7: descriptive catalogue, explicitly no completeness inference.
pc='iteration_5/data/processed/geonet_M5_shallow_1960_2026.csv'
if PORTABLE:
    catdata=json.loads((DATA/'catalogue.json').read_text()); labels=Counter(catdata['native_labels']); ncat=catdata['event_count']
else:
    cat=readcsv(pc); years=np.array([int(r['time_utc'][:4]) for r in cat]); mag=np.array([float(r['magnitude']) for r in cat]); labels=Counter(r['magnitude_type'] for r in cat); ncat=len(cat)
fig,axes=plt.subplots(1,3,figsize=(7.3,3.2),gridspec_kw={'width_ratios':[1.5,1,1]})
yy=np.arange(1960,2027); counts=np.array([catdata['annual_counts'][str(y)] for y in yy]) if PORTABLE else np.array([(years==y).sum() for y in yy]); axes[0].bar(yy,counts,width=1,color=BLUE); axes[0].set(xlabel='UTC year',ylabel='Selected events',title='Archived coverage'); axes[0].set_xticks([1960,1980,2000,2026]); axes[0].tick_params(axis='x',labelsize=8)
axes[1].stairs(catdata['magnitude_histogram_counts'],catdata['magnitude_histogram_edges'],fill=True,color=BLUE); axes[1].set(xlabel='Native magnitude',title='Selected magnitudes')
labs=list(sorted(labels,key=labels.get,reverse=True)); axes[2].barh([r'$M_{\mathrm{L}}$' if k == 'ML' else k for k in labs[::-1]],[labels[k] for k in labs[::-1]],color=GRAY); axes[2].set(xlabel='Selected events',title='Native scale labels'); axes[2].tick_params(axis='y',labelsize=8)
fig.tight_layout(w_pad=1)
save(fig,'catalogue','Descriptive archive coverage and native mark composition of 913 selected M>=5 shallow events. The final calendar year is partial; no completeness claim follows.',[pc],{'event_count':ncat,'annual_counts':dict(zip(map(str,yy),map(int,counts))),'native_labels':labels,'magnitude_histogram_counts':catdata['magnitude_histogram_counts'],'magnitude_histogram_edges':catdata['magnitude_histogram_edges'],'end_exclusive':'2026-10-15'})

# 8: exact algebraic nonclosure example, not a simulation.
g=math.log(2); fig,axes=plt.subplots(1,2,figsize=(7.1,3.6))
states=np.array([[0,g],[g/2,g/2]]); xx=np.arange(2)
axes[0].bar(xx-.16,states[:,0],width=.3,color=BLUE,label='Region 1'); axes[0].bar(xx+.16,states[:,1],width=.3,color=ORANGE,label='Region 2'); axes[0].set(xticks=xx,xticklabels=['Concentrated','Evenly split'],ylabel='Local inhibition state',title=r'Same pooled state: $G_1+G_2=\log 2$'); axes[0].legend(frameon=False,fontsize=9)
local=np.exp(-states).sum(axis=1); pooled=2*math.exp(-g)
axes[1].bar(xx,local,color=[BLUE,ORANGE],width=.5); axes[1].axhline(pooled,color=GRAY,ls='--',label=r'Pooled rule: $2B e^{-(G_1+G_2)}=B$')
for x,v,label in zip(xx,local,[r'$1.5$',r'$\sqrt{2}$']): axes[1].text(x,v+.025,label,ha='center',fontsize=12)
axes[1].set(xticks=xx,xticklabels=['Concentrated','Evenly split'],ylabel='Total background intensity / B',ylim=(0,1.8),title=r'Local rule: $B e^{-G_1}+B e^{-G_2}$'); axes[1].legend(frameon=False,fontsize=8,loc='lower center')
fig.tight_layout(w_pad=2)
save(fig,'pooling','Exact algebraic illustration with equal positive local backgrounds and eta=1: equal pooled states yield distinct local sums. Triggering is omitted; this is not simulated evidence.',[],{'type':'exact algebraic illustration','eta':1,'B1_over_B':1,'B2_over_B':1,'states':states.tolist(),'local_intensity_over_B':local.tolist(),'pooled_rule_over_B':pooled})
# 9: exact toy finite-mixture definitions, no fit or simulated process.
weights=np.array([.50,.35,.15]); rates=np.array([.05,1.,20.])
t=np.r_[0,np.geomspace(.001,100,400)]
phi=np.exp(-t[:,None]*rates)*weights
trigger=phi*rates
fig,axes=plt.subplots(1,2,figsize=(7.1,3.2))
for ax,parts,total,title in [(axes[0],phi,phi.sum(axis=1),r'Relaxation: $\Phi(t)=\sum_j w_j e^{-\vartheta_jt}$'),(axes[1],trigger,trigger.sum(axis=1),r'Triggering: $h(t)=-\Phi^{\prime}(t)$')]:
 for j,color in enumerate([BLUE,ORANGE,GRAY]): ax.plot(t,parts[:,j],color=color,lw=1,alpha=.7,label=f'Mode {j+1}')
 ax.plot(t,total,color='#222222',lw=2,label='Sum'); ax.set(xscale='symlog',xlim=(0,100),xlabel='Dimensionless lag',title=title); ax.set_xscale('symlog',linthresh=.01); ax.grid(alpha=.15)
axes[0].set_ylabel('Kernel value'); axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout(w_pad=1.4)
save(fig,'finite_memory','Exact illustrative three-mode curves with declared positive weights/rates. A toy algebraic mixture, not fitted parameters, a simulated catalogue, or the exact Mittag-Leffler kernel.',[],{'type':'exact toy finite-mixture illustration','weights':weights.tolist(),'dimensionless_rates':rates.tolist(),'lag':t.tolist(),'relaxation':phi.sum(axis=1).tolist(),'triggering':trigger.sum(axis=1).tolist()})
(FIG/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
print(f'Rendered {len(manifest)} figures from saved evidence and exact diagrams.')
