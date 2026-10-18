"""Descriptive frozen rolling-score/residual summaries; no calibration p-values."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import read_sealed,sha256

FAMILIES=['poisson_constant','poisson_loglinear','etas_omori','ml_trigger','ml_shared']

def residual_summary(increments,tail):
    z=np.asarray(increments,dtype=float);n=len(z)
    if not np.isfinite(z).all() or np.any(z<0) or not np.isfinite(tail) or tail<0:raise ValueError('invalid compensator increments')
    u=-np.expm1(-z);s=np.sort(u)
    ks=float(max(np.max(np.arange(1,n+1)/n-s),np.max(s-np.arange(n)/n))) if n else None
    centered=z-z.mean() if n else z;den=float(centered@centered)
    acf=[dict(lag=k,pairs=max(n-k,0),value=float(centered[k:]@centered[:-k]/den) if n>k and den>0 else None) for k in range(1,11)]
    return dict(n_complete_gaps=n,mean_compensator_gap=float(z.mean()) if n else None,
                uniform_KS_statistic=ks,acf=acf,terminal_censored_compensator=float(tail),
                realized_integrated_intensity=float(z.sum()+tail),
                count_minus_compensator=float(n-z.sum()-tail),
                interpretation='Descriptive: no fitted-KS p-value, count-Poisson test, or iid-fold uncertainty.')

def analyze(source,output,protocol,receipt):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    raw=json.loads(source.read_text());rows=[];residuals={};curves=[]
    protocol=Path(protocol);receipt=Path(receipt);p=read_sealed(protocol);r=read_sealed(receipt)
    if not (r['complete'] is True and r['exit_status']==0 and r['matching_generation_alive'] is False and r['charged_seconds']<=p['wall_cap_seconds']):raise ValueError('unsuccessful bounded generation')
    if source.resolve()!=(ROOT/p['execution']['output']).resolve() or receipt.resolve()!=(ROOT/p['execution']['receipt']).resolve():raise ValueError('unplanned analysis input')
    if raw['protocol_sha256']!=sha256(protocol) or raw['input_binding']['sha256']!=p['data']['sha256'] or sha256(ROOT/p['data']['path'])!=p['data']['sha256']:raise ValueError('protocol/data binding')
    if r['script_sha256']!=p['source_hashes']['iteration_27/code/run_adequacy.py'] or r['cap_seconds']!=p['wall_cap_seconds']:raise ValueError('execution binding')
    for path,h in p['source_hashes'].items():
        if sha256(ROOT/path)!=h:raise ValueError('frozen source changed')
    if any(p['source_hashes'].get(path)!=h for path,h in raw['source_hashes'].items()):raise ValueError('recorded source binding')
    if not raw['complete'] or raw['synthetic_only']:raise ValueError('requires completed observed evaluation')
    validate_observed(raw)
    import pandas as pd
    frame=pd.read_csv(ROOT/p['data']['path'])
    for f in raw['folds'].values():
        train=frame.t_days<f['train_end'];beta=float(train.sum()/frame.excess[train].sum())
        if abs(beta-f['fit']['beta_train'])>1e-12:raise ValueError('training-only beta')
        test=(frame.t_days>=f['train_end'])&(frame.t_days<f['test_end']);mark=float(test.sum()*np.log(beta)-beta*frame.excess[test].sum())
        for c in f['test_controls'].values():
            if abs(mark-c['order32']['common_mark_loglik'])>1e-8:raise ValueError('mark density join')
    for id,fold in raw['folds'].items():
        residuals[id]={}
        for family in FAMILIES:
            control=fold['test_controls'][family];q=control['order32'];selected=fold['fit']['best'][family]
            if not control['pass']:raise ValueError('failed numerical control')
            r=residual_summary(q['event_compensator_increments'],q['final_no_event_compensator'])
            if abs(r['realized_integrated_intensity']-q['integrated_intensity'])>1e-7:raise ValueError('gap/tail accounting')
            residuals[id][family]=r
            rows.append(dict(fold=id,model=family,n_train=fold['fit']['n_train'],n_test=q['n_events'],beta_train=fold['fit']['beta_train'],
                ground_loglik=q['ground_loglik'],common_mark_loglik=q['common_mark_loglik'],full_marked_loglik=q['full_marked_loglik'],
                integrated_intensity=q['integrated_intensity'],count_minus_compensator=q['count_minus_compensator'],
                selected_id=selected['id'],parameters=json.dumps(selected['pars']),ratio=selected['ratio'],kernel=json.dumps(selected['trigger']),
                uniform_KS_statistic=r['uniform_KS_statistic'],mean_gap=r['mean_compensator_gap']))
            for i,total in enumerate(np.cumsum(q['event_compensator_increments']),1):
                curves.append(dict(fold=id,model=family,event_index=i,cumulative_compensator=float(total),count_minus_compensator=float(i-total)))
    def csv_write(name,records):
        with open(output/name,'w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    csv_write('fold_scores.csv',rows);csv_write('compensator_curves.csv',curves)
    totals={family:{k:float(sum(r[k] for r in rows if r['model']==family)) for k in ['ground_loglik','common_mark_loglik','full_marked_loglik','n_test']} for family in FAMILIES}
    comparisons={}
    for first,second in [('ml_shared','ml_trigger'),('etas_omori','ml_trigger'),('etas_omori','ml_shared'),('ml_trigger','poisson_loglinear')]:
        differences={id:next(r['ground_loglik'] for r in rows if r['fold']==id and r['model']==first)-next(r['ground_loglik'] for r in rows if r['fold']==id and r['model']==second) for id in raw['folds']}
        comparisons[first+' minus '+second]=dict(by_fold=differences,total=sum(differences.values()),per_event=sum(differences.values())/totals[first]['n_test'])
    result=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),protocol_sha256=raw['protocol_sha256'],execution_receipt_sha256=sha256(receipt),
        complete=True,exploratory=True,totals=totals,comparisons=comparisons,residuals=residuals,
        controls=dict(max_abs_quadrature_delta=max(abs(c['quadrature_delta']) for f in raw['folds'].values() for k in ['training_controls','test_controls'] for c in f[k].values()),
                      max_abs_spectral_delta=max(abs(c['spectral_delta']) for f in raw['folds'].values() for k in ['training_controls','test_controls'] for c in f[k].values())),
        uncertainty='No inferential p-values or independent-fold standard errors. Same previously seen catalogue, overlapping training histories; no calibrated model-adequacy acceptance.',
        families=FAMILIES)
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,(id,fold) in zip(axes,raw['folds'].items()):
        for family in FAMILIES:
            u=np.sort(fold['test_controls'][family]['order32']['rescaled_uniform']);ax.step(u,np.arange(1,len(u)+1)/len(u),where='post',label=family)
        ax.plot([0,1],[0,1],color='black',ls=':',lw=1);ax.set(title=id,xlabel='Rescaled gap u',ylabel='Empirical CDF')
    axes[-1].legend(fontsize=7);fig.suptitle('Exploratory time-rescaling diagnostics (terminal gaps censored)');fig.savefig(output/'residual_ecdfs.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');base=totals['poisson_loglinear']['ground_loglik']
    ax.bar(FAMILIES,[totals[m]['ground_loglik']-base for m in FAMILIES]);ax.axhline(0,color='black',lw=.7)
    ax.tick_params(axis='x',labelrotation=20);ax.set(ylabel='Ground logscore difference',title='Sum over three disjoint test periods; loglinear Poisson reference')
    fig.savefig(output/'test_scores.png',dpi=150);plt.close(fig)
    return result

def validate_observed(raw):
    from adequacy import validate_fit
    specs=[('fold_50',.5,.65,301,203),('fold_65',.65,.8,504,155),('fold_80',.8,1.,659,254)]
    if list(raw['folds'])!=[s[0] for s in specs] or raw['T0']!=24394.:raise ValueError('frozen fold coverage/order')
    for id,train,end,ntrain,ntest in specs:
        f=raw['folds'][id]
        if f.get('complete') is not True or f['spec']!=dict(id=id,train_fraction=train,test_end_fraction=end) or f['fit']['n_train']!=ntrain:raise ValueError('incomplete/mismatched fold')
        validate_fit(f['fit'])
        for controls in ['test_controls','training_controls']:
            if set(f[controls])!=set(FAMILIES):raise ValueError('model coverage')
            for c in f[controls].values():
                if c['pass'] is not True or not(abs(c['quadrature_delta'])<1e-4 and abs(c['spectral_delta'])<1e-4):raise ValueError('numerical controls')
        for family in FAMILIES:
            q=f['test_controls'][family]['order32'];z=np.asarray(q['event_compensator_increments']);logs=np.asarray(q['event_log_intensities']);u=np.asarray(q['rescaled_uniform'])
            if not(q['n_events']==ntest==len(z)==len(logs)==len(u)) or not q['final_tail_right_censored']:raise ValueError('event quota/censoring')
            if not np.allclose(u,-np.expm1(-z),rtol=0,atol=1e-14):raise ValueError('stored uniforms differ')
            if abs(q['ground_loglik']-(logs.sum()-q['integrated_intensity']))>1e-7 or abs(q['full_marked_loglik']-q['ground_loglik']-q['common_mark_loglik'])>1e-7:raise ValueError('score accounting')
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--protocol',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();print(json.dumps(analyze(a.source,a.output,a.protocol,a.receipt)['comparisons'],indent=2))
