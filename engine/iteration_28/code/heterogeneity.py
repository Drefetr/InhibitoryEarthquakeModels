"""Fixed-recipe heterogeneity audit; no optimizer, simulator or download API.

Geographic/mark groups partition EVENT terms only. Full compensators belong to
the temporal observation window; no spatial marked-process law is introduced.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
AUTHORITY=ROOT/'iteration_27/code/adequacy.py'
AUTHORITY_SHA='9662c90f9ff7e120fe26c1dfc0bd969071f16774feec686b9e97731337e0ff90'
if hashlib.sha256(AUTHORITY.read_bytes()).hexdigest()!=AUTHORITY_SHA:raise ValueError('Frozen predictive authority changed')
_spec=importlib.util.spec_from_file_location('_heterogeneity28_adequacy',AUTHORITY)
_authority=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_authority;_spec.loader.exec_module(_authority)
COLUMNS=['event_id','time_utc','t_days','magnitude','excess','depth','latitude','longitude','magnitude_type']
AXES=('spatial_cell','magnitude_band','depth_band','native_label')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def audit_spec():return json.loads((ROOT/'iteration_28/configs/D_interface.json').read_text())['audit']
def deadline_check(deadline):
    if deadline is not None and time.perf_counter()>=deadline:raise TimeoutError('Frozen fixed-parameter audit deadline reached')


def frame_fingerprint(frame):
    return hashlib.sha256(frame[COLUMNS].to_csv(index=False,float_format='%.17g',lineterminator='\n').encode()).hexdigest()


def in_memory_bindings(frame,observed,fdsn_frame=None):
    """Content joins for loaded inputs; byte authority is supplied by loader."""
    out={'frame_sha256':frame_fingerprint(frame),'observed_object_sha256':canonical(observed)}
    if fdsn_frame is not None:out['fdsn_frame_sha256']=frame_fingerprint(fdsn_frame)
    return out


def load_bound_inputs(paths,expected_sha256):
    """Read only actual hash-bound files; all four inputs are mandatory.

    paths has catalogue,observed,observed_protocol,fdsn. expected_sha256 is
    keyed by the exact path strings in paths; root freezes those expected pins.
    """
    required={'catalogue','observed','observed_protocol','fdsn'}
    if set(paths)!=required:raise ValueError('input roles differ')
    actual={}
    for role,path in paths.items():
        full=Path(path);full=full if full.is_absolute() else ROOT/full
        if path not in expected_sha256 or sha(full)!=expected_sha256[path]:raise ValueError('input byte binding: '+role)
        actual[role]=full
    observed=json.loads(actual['observed'].read_text());old=json.loads(actual['observed_protocol'].read_text())
    if observed['protocol_sha256']!=sha(actual['observed_protocol']) or observed['input_binding']['sha256']!=sha(actual['catalogue']):raise ValueError('saved predictive input lineage')
    frame=pd.read_csv(actual['catalogue']);fdsn=pd.read_csv(actual['fdsn'])
    bindings=in_memory_bindings(frame,observed,fdsn);bindings['source_hashes']={path:sha(actual[role]) for role,path in paths.items()}
    return dict(frame=frame,observed=observed,observed_protocol=old,fdsn_frame=fdsn,bindings=bindings)


def _validate_frame(frame):
    if any(k not in frame for k in COLUMNS) or frame.event_id.astype(str).duplicated().any():raise ValueError('missing column or duplicate event identity')
    numeric=frame[['t_days','magnitude','excess','depth','latitude','longitude']].to_numpy(float)
    if not np.isfinite(numeric).all() or np.any(np.diff(frame.t_days.to_numpy(float))<=0) or np.any(frame.t_days<0):raise ValueError('invalid ordered catalogue')
    if np.any(frame.magnitude<5) or np.any(frame.excess<0) or np.max(np.abs((frame.magnitude-frame.excess-5).to_numpy()),initial=0)>1e-10:raise ValueError('magnitude/excess support')


def _saved_score(observed,fold,model,order):return observed['folds'][fold]['test_controls'][model][f'order{order}']


def join_saved_events(frame,observed,bindings):
    """Pure exact ordered join: no new event intensity or likelihood calls."""
    _validate_frame(frame)
    if frame_fingerprint(frame)!=bindings['frame_sha256'] or canonical(observed)!=bindings['observed_object_sha256']:raise ValueError('in-memory input changed')
    if observed.get('complete') is not True or observed.get('synthetic_only') is not False or observed.get('status')!='complete':raise ValueError('saved run incomplete')
    for key,col in [('times_f64le','t_days'),('excess_f64le','excess')]:
        digest=hashlib.sha256(np.ascontiguousarray(frame[col].to_numpy(),dtype='<f8').tobytes()).hexdigest()
        if observed['array_binding'][key]!=digest:raise ValueError('saved array ordering/marks changed')
    rows=[];seen=set();spec=audit_spec()
    for fold,f in observed['folds'].items():
        if f.get('complete') is not True or f.get('status')!='complete':raise ValueError('incomplete fold')
        indexes=np.flatnonzero((frame.t_days>=f['train_end'])&(frame.t_days<f['test_end']))
        for model in spec['models']:
            for order in spec['orders']:
                q=_saved_score(observed,fold,model,order)
                if q['n_events']!=len(indexes) or len(q['event_log_intensities'])!=len(indexes) or not np.isfinite(q['event_log_intensities']).all():raise ValueError('event vector quota')
                if q['start']!=f['train_end'] or q['end']!=f['test_end'] or q['order']!=order:raise ValueError('saved score window/order')
                if abs(math.fsum(q['event_log_intensities'])-q['integrated_intensity']-q['ground_loglik'])>spec['arithmetic_tolerance']:raise ValueError('saved event/score arithmetic')
        for ordinal,index in enumerate(indexes):
            row={k:(str(frame.iloc[index][k]) if k in ['event_id','time_utc','magnitude_type'] else float(frame.iloc[index][k])) for k in COLUMNS}
            if row['event_id'] in seen:raise ValueError('test event repeated across folds')
            seen.add(row['event_id']);row.update(fold=fold,fold_event_index=ordinal,catalogue_index=int(index))
            for model in spec['models']:
                for order in spec['orders']:row[f'log_intensity_{model}_{order}']=float(_saved_score(observed,fold,model,order)['event_log_intensities'][ordinal])
            for comparison in spec['comparisons']:
                for order in spec['orders']:row[f'event_logratio_{comparison["id"]}_{order}']=row[f'log_intensity_{comparison["model"]}_{order}']-row[f'log_intensity_{comparison["baseline"]}_{order}']
            rows.append(row)
    return rows


def _closed_axis(value,edges):
    if not np.isfinite(value) or value<edges[0] or value>edges[-1]:raise ValueError('spatial value outside frozen support')
    return min(int(np.searchsorted(edges,value,side='right')-1),len(edges)-2)


def categories(spec=None):
    spec=spec or audit_spec()
    return dict(spatial_cell=[f'lat{i}_lon{j}' for i in range(3) for j in range(3)],
        magnitude_band=['M5_5.5','M5.5_6','M6_6.5','M6.5_plus'],
        depth_band=['depth_le10','depth_10_20','depth_20_40'],native_label=spec['native_labels'])


def categorize(row,spec=None):
    spec=spec or audit_spec();mag=float(row['magnitude']);depth=float(row['depth']);label=str(row['magnitude_type'])
    if not np.isfinite([mag,depth]).all() or mag<5 or depth>40 or label not in spec['native_labels']:raise ValueError('metadata outside frozen categories')
    i=_closed_axis(float(row['latitude']),spec['latitude_edges']);j=_closed_axis(float(row['longitude']),spec['longitude_edges'])
    return dict(spatial_cell=f'lat{i}_lon{j}',magnitude_band=categories(spec)['magnitude_band'][min(int(np.searchsorted(spec['magnitude_edges'],mag,side='right')-1),3)],
                depth_band=categories(spec)['depth_band'][int(np.searchsorted(spec['depth_upper_edges'],depth,side='left'))],native_label=label)


def group_event_contributions(frame,events,observed,spec=None):
    spec=spec or audit_spec();cats=categories(spec);groups=[];composition=[];tv=[];global_rows=[]
    classified=[categorize(row,spec) for row in frame.to_dict('records')]
    for fold,f in observed['folds'].items():
        joined=[r for r in events if r['fold']==fold];train=np.flatnonzero(frame.t_days<f['train_end']);test=[r['catalogue_index'] for r in joined]
        if not len(train) or not len(test):raise ValueError('composition requires nonempty frozen train/test')
        for axis in AXES:
            total=0.;counts=0
            for category in cats[axis]:
                nt=sum(classified[i][axis]==category for i in train);ns=sum(classified[i][axis]==category for i in test);pt=nt/len(train);ps=ns/len(test)
                composition.append(dict(fold=fold,dimension=axis,category=category,n_train=nt,n_test=ns,p_train=pt,p_test=ps,sparse_test=ns<spec['sparse_count_below']))
                total+=abs(pt-ps);counts+=ns
                subset=[r for r in joined if classified[r['catalogue_index']][axis]==category]
                for comparison in spec['comparisons']:
                    values=[r[f'event_logratio_{comparison["id"]}_{spec["group_score_order"]}'] for r in subset];value=math.fsum(values)
                    groups.append(dict(fold=fold,comparison=comparison['id'],role=comparison['role'],dimension=axis,category=category,n_events=len(values),sparse=len(values)<spec['sparse_count_below'],event_logratio_sum=value,event_logratio_mean=value/len(values) if values else None,positive_events=sum(x>0 for x in values),negative_events=sum(x<0 for x in values),zero_events=sum(x==0 for x in values),component_scope='event component only; no category compensator or full local likelihood'))
            if counts!=len(test):raise ValueError('category conservation')
            tv.append(dict(fold=fold,dimension=axis,total_variation=.5*total,n_train=len(train),n_test=len(test)))
        for comparison in spec['comparisons']:
            for order in spec['orders']:
                first=_saved_score(observed,fold,comparison['model'],order);base=_saved_score(observed,fold,comparison['baseline'],order)
                E=math.fsum(np.asarray(first['event_log_intensities'])-np.asarray(base['event_log_intensities']));C=base['integrated_intensity']-first['integrated_intensity'];D=first['ground_loglik']-base['ground_loglik']
                if abs(E+C-D)>spec['arithmetic_tolerance'] or abs(first['common_mark_loglik']-base['common_mark_loglik'])>spec['arithmetic_tolerance']:raise ValueError('global score decomposition')
                global_rows.append(dict(fold=fold,comparison=comparison['id'],order=order,beta=f['fit']['beta_train'],event_logratio=E,global_compensator_contribution=C,ground_logscore_difference=D,full_marked_logscore_difference=first['full_marked_loglik']-base['full_marked_loglik'],arithmetic_residual=E+C-D,model_integrated_intensity=first['integrated_intensity'],baseline_integrated_intensity=base['integrated_intensity'],model_ground_loglik=first['ground_loglik'],baseline_ground_loglik=base['ground_loglik'],common_mark_loglik=first['common_mark_loglik']))
                if order==spec['group_score_order']:
                    for axis in AXES:
                        total=math.fsum(r['event_logratio_sum'] for r in groups if r['fold']==fold and r['comparison']==comparison['id'] and r['dimension']==axis)
                        if abs(total-E)>spec['arithmetic_tolerance']:raise ValueError('group event conservation')
    return dict(event_groups=groups,composition=composition,total_variation=tv,global_decomposition=global_rows)


def _evaluate(frame,observed,fold,model,start,end,order,deadline):
    deadline_check(deadline);f=observed['folds'][fold]
    score=_authority.evaluate_window(frame.t_days.to_numpy(),frame.excess.to_numpy(),f['fit']['best'][model],start,end,observed['T0'],f['fit']['beta_train'],order)
    deadline_check(deadline)
    return score


def temporal_decomposition(frame,observed,bindings,spec,deadline,progress_callback=None):
    join_saved_events(frame,observed,bindings);rows=[];comparisons=[];controls=[];additivity=[]
    for fold,f in observed['folds'].items():
        edges=[f['train_end']+(f['test_end']-f['train_end'])*k/spec['temporal_bins_per_fold'] for k in range(spec['temporal_bins_per_fold']+1)]
        for index,(start,end) in enumerate(zip(edges[:-1],edges[1:])):
            for model in spec['models']:
                pair=[]
                for order in spec['orders']:
                    score=_evaluate(frame,observed,fold,model,start,end,order,deadline)
                    row=dict(fold=fold,bin=index,model=model,order=order,start=start,end=end,selected_record_sha256=canonical(f['fit']['best'][model]),score=score);rows.append(row);pair.append(score)
                delta=pair[1]['ground_loglik']-pair[0]['ground_loglik'];passed=abs(delta)<spec['numerical_tolerance'];controls.append(dict(fold=fold,bin=index,model=model,order32_minus16=delta,passed=passed))
                if progress_callback:progress_callback('temporal_pair',dict(scores=rows[-2:],order_control=controls[-1]))
                if not passed:raise ValueError('temporal order control failed')
            for comp in spec['comparisons']:
                for order in spec['orders']:
                    first=next(r['score'] for r in rows if r['fold']==fold and r['bin']==index and r['model']==comp['model'] and r['order']==order);base=next(r['score'] for r in rows if r['fold']==fold and r['bin']==index and r['model']==comp['baseline'] and r['order']==order)
                    E=math.fsum(first['event_log_intensities'])-math.fsum(base['event_log_intensities']);C=base['integrated_intensity']-first['integrated_intensity'];D=first['ground_loglik']-base['ground_loglik']
                    if abs(E+C-D)>spec['arithmetic_tolerance']:raise ValueError('temporal arithmetic')
                    comparisons.append(dict(fold=fold,bin=index,comparison=comp['id'],role=comp['role'],order=order,event_logratio=E,compensator_contribution=C,ground_logscore_difference=D,full_marked_logscore_difference=first['full_marked_loglik']-base['full_marked_loglik'],arithmetic_residual=E+C-D))
        for model in spec['models']:
            for order in spec['orders']:
                subset=[r['score'] for r in rows if r['fold']==fold and r['model']==model and r['order']==order];saved=_saved_score(observed,fold,model,order)
                errors={key:math.fsum(r[key] for r in subset)-saved[key] for key in ['ground_loglik','full_marked_loglik','common_mark_loglik','integrated_intensity']}
                if max(abs(x) for x in errors.values())>=spec['numerical_tolerance'] or sum(r['n_events'] for r in subset)!=saved['n_events']:raise ValueError('temporal full-window additivity')
                additivity.append(dict(fold=fold,model=model,order=order,errors=errors,passed=True))
    return dict(scores=rows,comparisons=comparisons,order_controls=controls,additivity_controls=additivity,evaluations=len(rows))


def prepare_magnitude_values(frame,fdsn,bindings,spec=None):
    spec=spec or audit_spec();_validate_frame(frame)
    if fdsn.event_id.astype(str).duplicated().any():raise ValueError('duplicate alternate event identity')
    if frame_fingerprint(frame)!=bindings['frame_sha256'] or frame_fingerprint(fdsn)!=bindings['fdsn_frame_sha256']:raise ValueError('source branch input changed')
    ids=frame.event_id.astype(str);alternate=fdsn.copy();alternate['event_id']=alternate.event_id.astype(str)
    extra=set(alternate.event_id)-set(ids)
    if not extra.issubset(set(spec['allowed_extra_fdsn_ids'])) or not set(ids).issubset(set(alternate.event_id)):raise ValueError('source one-to-one ID support')
    aligned=alternate.set_index('event_id').loc[ids];magnitudes=aligned.magnitude.to_numpy(float)
    if not np.isfinite(magnitudes).all() or np.any(magnitudes<5):raise ValueError('alternate magnitude support')
    result=frame.copy();result['magnitude']=magnitudes;result['excess']=magnitudes-5
    return result,dict(n_primary=len(frame),n_matched=len(aligned),ignored_extra_ids=sorted(extra),aligned_alternate_magnitudes=magnitudes.tolist(),n_magnitude_values_changed=int(np.count_nonzero(magnitudes!=frame.magnitude.to_numpy(float))),maximum_absolute_magnitude_shift=float(np.max(np.abs(magnitudes-frame.magnitude.to_numpy(float)),initial=0)),scope='archived interface magnitude-value sensitivity; primary clocks/location/depth/labels and saved recipes fixed; not catalogue replacement or physical homogenization')


def magnitude_value_sensitivity(frame,fdsn_frame,observed,bindings,spec,deadline,progress_callback=None):
    join_saved_events(frame,observed,bindings);changed,join=prepare_magnitude_values(frame,fdsn_frame,bindings,spec);rows=[];comparisons=[];controls=[]
    for fold,f in observed['folds'].items():
        for model in spec['models']:
            pair=[]
            for order in spec['orders']:
                score=_evaluate(changed,observed,fold,model,f['train_end'],f['test_end'],order,deadline);row=dict(fold=fold,model=model,order=order,beta_fixed=f['fit']['beta_train'],selected_record_sha256=canonical(f['fit']['best'][model]),score=score);rows.append(row);pair.append(score)
            delta=pair[1]['ground_loglik']-pair[0]['ground_loglik'];passed=abs(delta)<spec['numerical_tolerance'];controls.append(dict(fold=fold,model=model,order32_minus16=delta,passed=passed))
            if progress_callback:progress_callback('magnitude_value_pair',dict(scores=rows[-2:],order_control=controls[-1]))
            if not passed:raise ValueError('source branch order control failed')
        for comp in spec['comparisons']:
            for order in spec['orders']:
                first=next(r['score'] for r in rows if r['fold']==fold and r['model']==comp['model'] and r['order']==order);base=next(r['score'] for r in rows if r['fold']==fold and r['model']==comp['baseline'] and r['order']==order)
                saved=_saved_score(observed,fold,comp['model'],order)['ground_loglik']-_saved_score(observed,fold,comp['baseline'],order)['ground_loglik'];value=first['ground_loglik']-base['ground_loglik']
                E=math.fsum(first['event_log_intensities'])-math.fsum(base['event_log_intensities']);C=base['integrated_intensity']-first['integrated_intensity']
                if abs(E+C-value)>spec['arithmetic_tolerance']:raise ValueError('source decomposition')
                comparisons.append(dict(fold=fold,comparison=comp['id'],role=comp['role'],order=order,event_logratio=E,global_compensator_contribution=C,ground_logscore_difference=value,full_marked_logscore_difference=first['full_marked_loglik']-base['full_marked_loglik'],primary_ground_logscore_difference=saved,signed_relative_score_shift=value-saved,descriptive_sign_reversal=bool(saved*value<0),arithmetic_residual=E+C-value))
    return dict(join=join,scores=rows,comparisons=comparisons,order_controls=controls,evaluations=len(rows))


def validate_result(payload,spec=None):
    """Pure receipt checks; never calls a predictive history or model scorer."""
    spec=spec or audit_spec()
    if payload.get('complete') is not True:raise ValueError('incomplete audit')
    if payload.get('audit_spec')!=spec:raise ValueError('audit specification binding')
    groups=payload['groups'];temporal=payload['temporal'];source=payload['magnitude_value_sensitivity'];events=payload['events'];cats=categories(spec)
    folds={'fold_50':(301,203,.5,.65),'fold_65':(504,155,.65,.8),'fold_80':(659,254,.8,1.)};T0=24394.
    A=spec['arithmetic_tolerance'];N=spec['numerical_tolerance'];models=spec['models'];orders=spec['orders'];comps={c['id']:c for c in spec['comparisons']}
    def near(a,b,tol=A):
        if not np.isfinite([a,b]).all() or abs(a-b)>tol:raise ValueError('nonfinite or inconsistent saved arithmetic')
    def keyed(rows,fields,expected):
        mapping={tuple(r[k] for k in fields):r for r in rows}
        if len(mapping)!=len(rows) or set(mapping)!=set(expected):raise ValueError('missing/duplicate/unexpected output key')
        return mapping
    if len(events)!=612 or len(set(r['event_id'] for r in events))!=612:raise ValueError('unique test-event quota')
    if [(r['fold'],r['fold_event_index'],r['catalogue_index']) for r in events]!=[(f,i,train+i) for f,(train,n,_,__) in folds.items() for i in range(n)]:raise ValueError('ordered event identity join')
    classified={}
    for f,(ntrain,n,a,b) in folds.items():
        subset=[r for r in events if r['fold']==f]
        if not all(a*T0<=r['t_days']<b*T0 for r in subset) or np.any(np.diff([r['t_days'] for r in subset])<=0):raise ValueError('event window/order')
        for row in subset:
            classified[row['event_id']]=categorize(row,spec)
            near(row['magnitude']-5,row['excess'],1e-10)
            for c in comps.values():
                for order in orders:near(row[f'event_logratio_{c["id"]}_{order}'],row[f'log_intensity_{c["model"]}_{order}']-row[f'log_intensity_{c["baseline"]}_{order}'])
    expected_group=[(f,c,axis,cat) for f in folds for c in comps for axis in AXES for cat in cats[axis]]
    group_map=keyed(groups['event_groups'],['fold','comparison','dimension','category'],expected_group)
    composition=keyed(groups['composition'],['fold','dimension','category'],[(f,axis,cat) for f in folds for axis in AXES for cat in cats[axis]])
    tv=keyed(groups['total_variation'],['fold','dimension'],[(f,axis) for f in folds for axis in AXES])
    for (f,c,axis,cat),row in group_map.items():
        if set(row)!={'fold','comparison','role','dimension','category','n_events','sparse','event_logratio_sum','event_logratio_mean','positive_events','negative_events','zero_events','component_scope'}:raise ValueError('unexpected category field; no local compensator law')
        if any('compensator' in str(k).lower() or str(k).lower().startswith(('lambda','cell_comp','local_comp')) for k in row):raise ValueError('unjustified category compensator')
        values=[e[f'event_logratio_{c}_{spec["group_score_order"]}'] for e in events if e['fold']==f and classified[e['event_id']][axis]==cat];n=len(values);total=math.fsum(values)
        if row['n_events']!=n or row['sparse']!=(n<spec['sparse_count_below']) or row['positive_events']!=sum(x>0 for x in values) or row['negative_events']!=sum(x<0 for x in values) or row['zero_events']!=sum(x==0 for x in values) or row['role']!=comps[c]['role']:raise ValueError('group counts/policy')
        near(row['event_logratio_sum'],total)
        if n:near(row['event_logratio_mean'],total/n)
        elif row['event_logratio_mean'] is not None or row['event_logratio_sum']!=0:raise ValueError('empty category policy')
    for f,(ntrain,ntest,_,__) in folds.items():
        for axis in AXES:
            part=[composition[(f,axis,cat)] for cat in cats[axis]]
            if sum(r['n_train'] for r in part)!=ntrain or sum(r['n_test'] for r in part)!=ntest:raise ValueError('composition totals')
            for row in part:
                n=sum(e['fold']==f and classified[e['event_id']][axis]==row['category'] for e in events)
                if row['n_test']!=n or row['n_train']<0 or row['sparse_test']!=(n<spec['sparse_count_below']):raise ValueError('composition category')
                near(row['p_train'],row['n_train']/ntrain);near(row['p_test'],n/ntest)
            near(tv[(f,axis)]['total_variation'],.5*math.fsum(abs(r['p_test']-r['p_train']) for r in part))
            if tv[(f,axis)]['n_train']!=ntrain or tv[(f,axis)]['n_test']!=ntest:raise ValueError('TV sample quota')
    global_map=keyed(groups['global_decomposition'],['fold','comparison','order'],[(f,c,o) for f in folds for c in comps for o in orders]);absolute={}
    for (f,c,o),row in global_map.items():
        comp=comps[c];subset=[e for e in events if e['fold']==f];E=math.fsum(e[f'event_logratio_{c}_{o}'] for e in subset);C=row['baseline_integrated_intensity']-row['model_integrated_intensity'];D=row['model_ground_loglik']-row['baseline_ground_loglik']
        near(row['event_logratio'],E);near(row['global_compensator_contribution'],C);near(row['ground_logscore_difference'],D);near(row['full_marked_logscore_difference'],D);near(row['arithmetic_residual'],E+C-D);near(E+C,D)
        beta=row['beta']
        if not np.isfinite(beta) or beta<=0:raise ValueError('mark parameter')
        mark=len(subset)*math.log(beta)-beta*math.fsum(e['excess'] for e in subset);near(row['common_mark_loglik'],mark)
        for model,prefix in [(comp['model'],'model'),(comp['baseline'],'baseline')]:
            value=dict(integrated_intensity=row[prefix+'_integrated_intensity'],ground_loglik=row[prefix+'_ground_loglik'],common_mark_loglik=mark,full_marked_loglik=row[prefix+'_ground_loglik']+mark,beta=beta)
            near(math.fsum(e[f'log_intensity_{model}_{o}'] for e in subset)-value['integrated_intensity'],value['ground_loglik'])
            if (f,model,o) in absolute:
                for k in value:near(absolute[(f,model,o)][k],value[k])
            absolute[(f,model,o)]=value
    def check_score(score,f,order,left,right,source_marks=None,primary_model=None):
        expected=[e for e in events if e['fold']==f and left<=e['t_days']<right];n=len(expected)
        if score['start']!=left or score['end']!=right or score['order']!=order or score['n_events']!=n or not score['final_tail_right_censored']:raise ValueError('score window/quota/censoring')
        z=np.asarray(score['event_compensator_increments']);logs=np.asarray(score['event_log_intensities']);u=np.asarray(score['rescaled_uniform']);tail=score['final_no_event_compensator']
        if len(z)!=n or len(logs)!=n or len(u)!=n or not np.isfinite(z).all() or np.any(z<0) or not np.isfinite(tail) or tail<0:raise ValueError('invalid score arrays')
        if not np.isfinite(logs).all() or not np.allclose(u,-np.expm1(-z),atol=1e-14,rtol=0):raise ValueError('rescaling')
        near(math.fsum(z)+tail,score['integrated_intensity']);near(math.fsum(logs)-score['integrated_intensity'],score['ground_loglik']);near(score['ground_loglik']+score['common_mark_loglik'],score['full_marked_loglik']);near(n-score['integrated_intensity'],score['count_minus_compensator'])
        beta=absolute[(f,models[0],order)]['beta'];qs=[e['excess'] if source_marks is None else source_marks[e['catalogue_index']]-5 for e in expected];near(score['common_mark_loglik'],n*math.log(beta)-beta*math.fsum(qs))
        if primary_model is not None:
            if any(abs(log-e[f'log_intensity_{primary_model}_{order}'])>A for log,e in zip(logs,expected)):raise ValueError('bin reset event histories')
        return expected
    bins=range(spec['temporal_bins_per_fold']);tkeys=[(f,k,m,o) for f in folds for k in bins for m in models for o in orders]
    temporal_map=keyed(temporal['scores'],['fold','bin','model','order'],tkeys)
    if temporal['evaluations']!=len(tkeys):raise ValueError('temporal evaluation count')
    selected={}
    for (f,k,m,o),row in temporal_map.items():
        _,__,a,b=folds[f];edges=[a*T0+(b*T0-a*T0)*j/spec['temporal_bins_per_fold'] for j in range(spec['temporal_bins_per_fold']+1)];left,right=edges[k:k+2]
        if row['start']!=left or row['end']!=right:raise ValueError('bin geometry')
        check_score(row['score'],f,o,left,right,primary_model=m)
        if (f,m) in selected and selected[(f,m)]!=row['selected_record_sha256']:raise ValueError('selected fixed recipe changed')
        selected[(f,m)]=row['selected_record_sha256']
    tc=keyed(temporal['order_controls'],['fold','bin','model'],[(f,k,m) for f in folds for k in bins for m in models])
    for (f,k,m),row in tc.items():
        delta=temporal_map[(f,k,m,32)]['score']['ground_loglik']-temporal_map[(f,k,m,16)]['score']['ground_loglik'];near(row['order32_minus16'],delta)
        if row['passed'] is not True or abs(delta)>=N:raise ValueError('temporal order control')
    add=keyed(temporal['additivity_controls'],['fold','model','order'],[(f,m,o) for f in folds for m in models for o in orders])
    for (f,m,o),row in add.items():
        if row['passed'] is not True or set(row['errors'])!={'ground_loglik','full_marked_loglik','common_mark_loglik','integrated_intensity'}:raise ValueError('additivity metadata')
        for key,value in row['errors'].items():
            error=math.fsum(temporal_map[(f,k,m,o)]['score'][key] for k in bins)-absolute[(f,m,o)][key];near(value,error)
            if abs(error)>=N:raise ValueError('temporal additivity')
    def comparisons_check(branch,key_fields,expected_keys,scoremap,is_source=False):
        mapping=keyed(branch['comparisons'],key_fields,expected_keys)
        for key,row in mapping.items():
            if is_source:f,c,o=key;k=None
            else:f,k,c,o=key
            comp=comps[c];first=scoremap[(f,comp['model'],o) if is_source else (f,k,comp['model'],o)]['score'];base=scoremap[(f,comp['baseline'],o) if is_source else (f,k,comp['baseline'],o)]['score']
            E=math.fsum(first['event_log_intensities'])-math.fsum(base['event_log_intensities']);C=base['integrated_intensity']-first['integrated_intensity'];D=first['ground_loglik']-base['ground_loglik']
            near(row['event_logratio'],E);near(row['global_compensator_contribution' if is_source else 'compensator_contribution'],C);near(row['ground_logscore_difference'],D);near(row['full_marked_logscore_difference'],first['full_marked_loglik']-base['full_marked_loglik']);near(row['arithmetic_residual'],E+C-D);near(E+C,D)
            if row['role']!=comp['role']:raise ValueError('comparison role')
            if is_source:
                old=global_map[(f,c,o)]['ground_logscore_difference'];near(row['primary_ground_logscore_difference'],old);near(row['signed_relative_score_shift'],D-old)
                if row['descriptive_sign_reversal']!=(old*D<0):raise ValueError('sign metadata')
    comparisons_check(temporal,['fold','bin','comparison','order'],[(f,k,c,o) for f in folds for k in bins for c in comps for o in orders],temporal_map)
    mags=source['join']['aligned_alternate_magnitudes']
    if source['join']['n_primary']!=913 or source['join']['n_matched']!=913 or len(mags)!=913 or not np.isfinite(mags).all() or min(mags)<5 or not set(source['join']['ignored_extra_ids']).issubset(set(spec['allowed_extra_fdsn_ids'])):raise ValueError('counterfactual support/join')
    source_map=keyed(source['scores'],['fold','model','order'],[(f,m,o) for f in folds for m in models for o in orders])
    if source['evaluations']!=18:raise ValueError('source evaluation quota')
    for (f,m,o),row in source_map.items():
        _,__,a,b=folds[f];check_score(row['score'],f,o,a*T0,b*T0,source_marks=mags)
        if row['selected_record_sha256']!=selected[(f,m)]:raise ValueError('source branch refitted recipe')
        near(row['beta_fixed'],absolute[(f,m,o)]['beta'])
    sc=keyed(source['order_controls'],['fold','model'],[(f,m) for f in folds for m in models])
    for (f,m),row in sc.items():
        delta=source_map[(f,m,32)]['score']['ground_loglik']-source_map[(f,m,16)]['score']['ground_loglik'];near(row['order32_minus16'],delta)
        if row['passed'] is not True or abs(delta)>=N:raise ValueError('source order control')
    comparisons_check(source,['fold','comparison','order'],[(f,c,o) for f in folds for c in comps for o in orders],source_map,True)
    return True
