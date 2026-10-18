"""Uniform-seed finite level-only fit; no saved-history work on import."""
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import math
import sys
import time
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
PROFILE_SHA='f0ab547015736eb9bfd6a3f3b8d08e60420516fda91a828149e5a5f087d9aff8'
if hashlib.sha256((ROOT/'iteration_24/code/profile_optimizer.py').read_bytes()).hexdigest()!=PROFILE_SHA:
    raise ValueError('Pinned ProfileObjective changed')
sys.path.insert(0,str(ROOT/'iteration_24/code'))
import profile_optimizer as profile
if Path(profile.__file__).resolve()!=(ROOT/'iteration_24/code/profile_optimizer.py').resolve():
    raise ValueError('Unexpected profile objective module')
ProfileObjective=profile.ProfileObjective
RestrictedHistory=profile.RestrictedHistory
CudaOracleContext=profile.CudaOracleContext
law_from_setting=profile.law_from_setting
clean=profile.clean
TOL=1e-4
SEARCH_SPEC=dict(a_bounds=[-12.,0.],eta_bounds=[0.,3.],supplied_d=True,
    amplitude_xtol=2e-12,amplitude_rtol=4*np.finfo(float).eps,amplitude_maxiter=100,
    eta_grid_points=17,scalar_extras=['saved_known_known_alternative_eta','zero'],
    scalar_xatol=1e-7,scalar_maxiter=100,
    local_bracket_rule='center<=both_neighbors_and_strictly_less_than_at_least_one;skip_fully_flat',
    eta_profile_points=11,eta_profiles_mandatory=True,
    eligible_floors='saved_known_known_alternative_then_null_then_own_new_null',
    baseline_case_template='iteration_21/results/full_staged/cases/{id}.json',
    baseline_authority='uniform_iteration21_known_known_for_all2149',
    seed_authority_change_from_iteration25='iteration21_uniform_instead_of_iteration24_fixed_ten',
    failed_solver_return_eligible=False,all_mandated_solvers_must_succeed=True,
    modal_query_chunk_nodes=32768,scalar_reduction_chunk_nodes=2**20,
    CUDA16_controls=8,CPU32_selected_controls=4,numerical_tolerance=TOL,identity_arithmetic_tolerance=1e-8,
    calibration_statistic_order=16,numerical_envelope_floor=1e-6,
    numerical_envelope_rule='max(1e-6,2*sum_phase_abs(fresh_CPU32_NLL-original_backend_CUDA16_NLL))',
    numerical_envelope_interpretation='empirical_sensitivity_interval_not_rigorous_error_bound')


def require(ok,message):
    if not ok:raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(clean(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def expected_law(protocol):
    c=protocol['law']
    return law_from_setting(dict(beta=c['beta'],horizon=c['horizon_days'],coordinate_horizon=c['coordinate_horizon_days'],generating_fit=c['alternative_generating_fit']))


def canonical_clock(mode):
    require(mode in ('projected','original_projected_clock','paired','paired_clock'),'Unrecognized explicit clock')
    return 'original_projected_clock' if mode in ('projected','original_projected_clock') else 'paired_clock'


def baseline_kk(case,law):
    p=case.get('payload',case)
    require(p.get('complete') is True and [m['id'] for m in p['methods']]==['known_known','background_estimated'],
            'Uniform complete two-method baseline required')
    if 'status' in case:require(case['status']=='complete','Baseline case incomplete')
    m=p['methods'][0]
    require(m.get('complete') is True and m.get('status')=='complete','Validated baseline KK method required')
    for phase in ('null','alternative'):
        r=m[phase];x=r['pars']
        require(len(x)==4 and all(math.isfinite(v) for v in x) and x[:3]==[law.a,law.d,law.n] and
                0<=x[3]<=3 and math.isfinite(r['nll']) and r['trigger']==asdict(law.trigger) and r['ratio']==law.ratio,
                'Baseline KK fixed law/coordinates differ')
    require(m['null']['pars'][3]==0.,'Baseline null eta differs')
    require('contrast16' in m and 'contrast32' in m,'Both saved baseline contrasts required')
    return m


def scalar_search(value,extra,deadline,records):
    seen=[]
    def capture(eta):
        p=value(eta);seen.append(copy.deepcopy(p));return p
    grid=np.linspace(0.,3.,17);runs=[]
    selected,eligible,stage=profile.scalar_search(capture,grid,extra,label='level_eta',deadline=deadline,records=runs)
    stage.update(grid=list(map(float,grid)),extra=list(map(float,extra)),grid_candidates=seen[:17],extra_candidates=seen[17:19],
                 selected=selected,eligible=eligible,runs=runs)
    records.extend(runs)
    return stage


def fit_level_case(arrays,law,clock_mode,protocol,baseline_case,context,deadline,progress_callback=None):
    require(protocol.get('search')==clean(SEARCH_SPEC),'Exact frozen level-only search required')
    require(asdict(law)==asdict(expected_law(protocol)),'Supplied full law differs')
    old=baseline_kk(baseline_case,law)
    input_binding=profile.authority.input_binding(arrays,law,clock_mode,protocol)
    prior_binding=old['input_binding']
    require(prior_binding['arrays']==input_binding['arrays'] and prior_binding['law']==input_binding['law'] and
            canonical_clock(prior_binding['clock_mode'])==canonical_clock(clock_mode),'Baseline source arrays/law/clock differ')
    if 'clock_mode' in baseline_case:require(canonical_clock(baseline_case['clock_mode'])==canonical_clock(clock_mode),'Baseline case clock differs')
    began=time.perf_counter();history=independent=objective=None;records=[];queries=[]
    reserve=protocol['resources']['validation_reserve_seconds'].get(baseline_case.get('id'),45.)
    search_deadline=deadline-reserve
    out=dict(schema_version=1,algorithm='uniform_seed_level_only',complete=False,status='partial',methods=[],
             search_records=records,queries=queries,sources={'iteration_24/code/profile_optimizer.py':PROFILE_SHA},
             baseline_binding=dict(id=baseline_case.get('id'),plan_id=baseline_case.get('plan_id'),canonical_case_sha256=digest(baseline_case),
                                   authority=SEARCH_SPEC['baseline_authority'],
                                   saved_input_binding={k:prior_binding[k] for k in ('arrays','law','clock_mode')}),
             input_binding=input_binding)
    def emit(value):
        if progress_callback is not None:progress_callback(clean(value))
    try:
        profile.cpu._deadline(search_deadline);context.begin_measurement()
        history=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=16,backend='cuda',context=context,deadline=search_deadline,
                                  chunk_nodes=32768,host_limit_bytes=2*2**30)
        objective=ProfileObjective(history,context=context,deadline=search_deadline,scalar_chunk_nodes=2**20)
        mark=history.mark_nll_constant
        out.update(mark_nll_constant=mark,trigger_compensator=objective.trigger_compensator)
        def fixed(a,eta):
            p=objective.fixed(a,law.d,eta,origin='fixed_amplitude');p['full_marked_nll']=p['nll']+mark;return p
        def level(eta):
            p=objective.evaluate(law.d,float(eta));p['full_marked_nll']=p['nll']+mark
            queries.append(dict(eta=float(eta),pars=p['pars'],nll=p['nll'],evaluation_id=p['evaluation_id']))
            return p
        kk=dict(id='known_known',saved_contrast16=old['contrast16'],saved_contrast32=old['contrast32'])
        for phase in ('null','alternative'):
            r=old[phase];kk[phase]=fixed(r['pars'][0],r['pars'][3]);kk['saved_'+phase+'_nll']=r['nll']
        out['baseline_method']=kk
        null_candidates=[kk['null'],level(0.)];null=min(null_candidates,key=lambda p:p['nll'])
        stage=scalar_search(level,[kk['alternative']['pars'][3],0.],search_deadline,records)
        curves=[]
        for i,eta in enumerate(np.linspace(0.,3.,11)):
            curves.append(dict(eta=float(eta),selected=null if i==0 else level(float(eta)),reused_null=i==0))
            emit(dict(event='eta_profile_completed',index=i,record=curves[-1]))
        candidates=[kk['alternative'],kk['null'],null]+stage['eligible']+[c['selected'] for c in curves]
        method=dict(id='level_only',null=null,alternative=min(candidates,key=lambda p:p['nll']),null_candidates=null_candidates,
                    candidates=candidates,search=stage,eta_profile=curves,complete=False)
        out['methods'].append(method);emit(dict(event='search_completed',selected=method['alternative'],solver_runs=len(records)))
        history.deadline=objective.deadline=deadline
        points=[('old_known_known_'+phase,kk[phase]) for phase in ('null','alternative')]
        points += [('new_level_only_'+phase,method[phase]) for phase in ('null','alternative')]
        points += [('level_corner_a_'+str(a)+'_eta_'+str(e),fixed(a,e)) for a in (-12.,0.) for e in (0.,3.)]
        controls=[]
        for label,p in points:
            v=history.objective(p['pars'],law.trigger,law.ratio,False);delta=abs(v-p['nll'])
            row=dict(label=label,pars=p['pars'],profile16_nll=p['nll'],old_backend16_nll=v,absolute_difference=delta,passed=delta<TOL)
            if label.startswith('old_'):
                phase='null' if label.endswith('_null') else 'alternative';saved=kk['saved_'+phase+'_nll']
                row.update(saved_primary16_nll=saved,saved_absolute_difference=abs(v-saved))
                row['passed']=bool(row['passed'] and abs(v-saved)<TOL and abs(p['nll']-saved)<TOL)
            controls.append(row)
        out['old_backend_controls']=controls
        for m in (kk,method):m['contrast16']=history.contrast(m['null'],m['alternative'])
        out['cuda_metrics']=context.cuda_metrics(32768)
        out['feature_metadata']=dict(clock_mode=history.mode,nodes16=history.nodes,preparation16_seconds=history.preparation_seconds,
                                      geometry_authority='pinned RestrictedHistory',shared_geometry_limitation='Order checks share history geometry; empirical controls are not rigorous universal bounds')
        history.close();history=None;context.torch.cuda.empty_cache()
        independent=RestrictedHistory(arrays,law,clock_mode=clock_mode,order=32,backend='cpu',deadline=deadline,chunk_nodes=32768,host_limit_bytes=2*2**30)
        require(independent.mark_nll_constant==mark,'Mark constant changed across orders')
        cpurows=[]
        for tag,m in [('old',kk),('new',method)]:
            for phase in ('null','alternative'):
                p=m[phase];v=independent.objective(p['pars'],law.trigger,law.ratio,False);delta=abs(v-p['nll'])
                r=dict(label=tag+'_'+m['id']+'_'+phase,phase=phase,pars=p['pars'],profile16_nll=p['nll'],independent32_nll=v,
                       absolute_difference=delta,passed=delta<TOL)
                if tag=='old':
                    r.update(saved_primary16_nll=m['saved_'+phase+'_nll'],saved_absolute_difference=abs(v-m['saved_'+phase+'_nll']))
                    r['passed']=bool(r['passed'] and r['saved_absolute_difference']<TOL)
                cpurows.append(r)
            m['contrast32']=independent.contrast(m['null'],m['alternative'])
        out['cpu32_controls']=cpurows
        cm={r['label']:r for r in controls};cp={r['label']:r for r in cpurows}
        selected=[]
        for phase in ('null','alternative'):
            label='new_level_only_'+phase;v16=cm[label]['old_backend16_nll'];v32=cp[label]['independent32_nll'];delta=abs(v32-v16)
            selected.append(dict(phase=phase,cuda16_nll=v16,cpu32_nll=v32,absolute_difference=delta,passed=delta<TOL))
        method['selected_objectives']=selected
        epsilon=max(1e-6,2*sum(r['absolute_difference'] for r in selected));statistic=method['contrast16']['statistic']
        out['calibration_score']=dict(statistic=statistic,numerical_envelope=epsilon,interval=[statistic-epsilon,statistic+epsilon],
                                      central_order=16,epsilon_floor=1e-6,
                                      interpretation=SEARCH_SPEC['numerical_envelope_interpretation'])
        out['feature_metadata'].update(nodes32=independent.nodes,preparation32_seconds=independent.preparation_seconds)
        identities=[]
        for order in (16,32):
            rows={r['label']:r for r in (controls if order==16 else cpurows)};field='old_backend16_nll' if order==16 else 'independent32_nll'
            g0,g1=(rows['old_known_known_'+phase][field]-rows['new_level_only_'+phase][field] for phase in ('null','alternative'))
            a,b=method['contrast'+str(order)],kk['contrast'+str(order)]
            lhs=a['statistic']-b['statistic'];rhs=2*(g1-g0)+a['direct_minus_subtraction']-b['direct_minus_subtraction']
            identities.append(dict(order=order,null_gain=g0,alternative_gain=g1,direct_statistic_change=lhs,
                                   direct_statistic_contraction_from_KK=-lhs,identity_rhs=rhs,residual=lhs-rhs))
        out['common_backend_identities']=identities
        out['baseline_binding']['compact_selected_sha256']=digest(kk)
        direct_ok=all(abs(m['contrast16']['statistic']-m['contrast32']['statistic'])<TOL and
                      all(abs(m['contrast'+str(o)]['direct_minus_subtraction'])<TOL and m['contrast'+str(o)]['statistic']>=-TOL for o in (16,32)) for m in (kk,method))
        saved_ok=all(abs(kk['contrast'+str(o)]['statistic']-kk['saved_contrast'+str(o)]['statistic'])<TOL for o in (16,32))
        floors_ok=all(method[phase]['nll']<=kk[phase]['nll']+1e-8 for phase in ('null','alternative'))
        nested=method['alternative']['nll']<=method['null']['nll']+1e-8
        solvers_ok=all(r['success'] for r in records)
        complete=bool(direct_ok and saved_ok and floors_ok and nested and solvers_ok and all(r['passed'] for r in controls+cpurows+selected) and
                      all(abs(r['residual'])<1e-8 for r in identities))
        method['complete']=complete
        method['status']='complete' if complete else 'unresolved_search_or_validation'
        out.update(complete=complete,status='complete' if complete else 'unresolved_search_or_validation',all_mandated_solvers_successful=solvers_ok,
                   direct_controls_passed=direct_ok,saved_baseline_controls_passed=saved_ok,validation_reserve_seconds=reserve)
        emit(dict(event='validation_completed',complete=complete))
    except Exception as exc:
        out.update(complete=False,status='time_cap' if isinstance(exc,TimeoutError) else 'unresolved',error=type(exc).__name__+': '+str(exc))
    finally:
        if objective is not None:
            out.update(evaluations=objective.evaluations,fixed_evaluations=objective.fixed_evaluations,evaluation_errors=objective.errors,
                       background_integral_evaluations=objective.J_evaluations,profile_requests=objective.profile_requests,profile_cache_hits=objective.cache_hits)
        for h in (history,independent):
            if h is not None:h.close()
        out['elapsed_seconds']=time.perf_counter()-began
    return clean(out)


def validate_result(payload,protocol):
    """Pure metadata/arithmetic guard; no history, likelihood or solver call."""
    require(protocol['search']==clean(SEARCH_SPEC),'Search protocol differs')
    require(payload.get('complete') is True and payload.get('status')=='complete' and payload['algorithm']=='uniform_seed_level_only','Incomplete level-only fit')
    require([m['id'] for m in payload['methods']]==['level_only'],'Only level-only is a new procedure')
    fit=protocol['law']['alternative_generating_fit'];a0,d0,n=fit['pars'][:3];mark=payload['mark_nll_constant'];binding=payload['input_binding']
    require(binding['contract_sha256']==digest(protocol) and binding['law']==clean(asdict(expected_law(protocol))),'Input law/contract binding differs')
    require(payload['baseline_binding']['authority']==SEARCH_SPEC['baseline_authority'],'Uniform baseline authority differs')
    prior=payload['baseline_binding']['saved_input_binding']
    require(prior['arrays']==binding['arrays'] and prior['law']==binding['law'] and canonical_clock(prior['clock_mode'])==canonical_clock(binding['clock_mode']),'Baseline input linkage differs')
    require(math.isfinite(mark),'Nonfinite mark constant')
    raw=payload['evaluations']
    def close(a,b,tol=1e-10):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol
    def point(p):
        x=p['pars']
        require(len(x)==4 and all(math.isfinite(v) for v in x) and -12<=x[0]<=0 and x[1:3]==[d0,n] and 0<=x[3]<=3 and
                p['trigger']==fit['trigger'] and p['ratio']==fit['ratio'] and math.isfinite(p['nll']),'Point outside fixed level-only model')
        require(close(p['full_marked_nll'],p['nll']+mark),'Marked objective differs')
        if p.get('origin')=='amplitude_profile':
            index=p['evaluation_id'];require(isinstance(index,int) and 0<=index<len(raw),'Profile reference missing')
            require(all(p[k]==raw[index][k] for k in ('pars','nll','amplitude','J','origin')),'Candidate detached from raw profile')
    for i,p in enumerate(raw):
        root=p['amplitude'];x=p['pars']
        require(p['evaluation_id']==i and x[1:3]==[d0,n] and 0<=x[3]<=3 and p['trigger']==fit['trigger'] and p['ratio']==fit['ratio'],'Raw profile law changed')
        require(root['root_converged'] is True and 0<=root['iterations']<=100 and -12<=root['a']<=0 and root['a']==x[0] and
                close(root['A'],math.exp(root['a']),1e-12) and root['n_events']==binding['arrays']['times']['shape'][0] and
                math.isfinite(p['J']) and p['J']>0 and all(math.isfinite(root[k]) for k in ('F_A_lower','F_A_upper','F_A_at_solution')),'Failed amplitude root metadata')
        rows=root['root_evaluations']
        require(len(rows)>=3 and rows[0]['a']==-12. and rows[1]['a']==0. and rows[-1]['a']==root['a'] and
                all(-12<=r['a']<=0 and math.isfinite(r['F_A']) for r in rows) and
                [rows[0]['F_A'],rows[1]['F_A'],rows[-1]['F_A']]==[root['F_A_lower'],root['F_A_upper'],root['F_A_at_solution']],'Root trace differs')
        b=root['boundary'];require((b=='lower' and x[0]==-12. and root['F_A_lower']>=0 and root['iterations']==0) or
                                  (b=='upper' and x[0]==0. and root['F_A_upper']<=0 and root['iterations']==0) or
                                  (b is None and -12<x[0]<0 and root['F_A_lower']<0<root['F_A_upper']),'Root boundary/end signs differ')
        require(close(p['nll'],root['objective_without_trigger']+payload['trigger_compensator'],1e-8),'Amplitude likelihood decomposition differs')
    m=payload['methods'][0];kk=payload['baseline_method'];require(kk['id']=='known_known','Baseline procedure differs')
    require(digest(kk)==payload['baseline_binding']['compact_selected_sha256'],'Compact baseline binding differs')
    for phase in ('null','alternative'):
        point(kk[phase]);require(kk[phase]['pars'][:3]==[a0,d0,n] and abs(kk[phase]['nll']-kk['saved_'+phase+'_nll'])<TOL,'Saved KK floor law/value differs')
    require(kk['null']['pars'][3]==0.,'Old null eta changed')
    st=m['search'];grid=list(np.linspace(0.,3.,17));extra=[kk['alternative']['pars'][3],0.]
    require(st['grid']==grid and st['extra']==extra and len(st['grid_candidates'])==17 and len(st['extra_candidates'])==2,'Mandatory scalar grid/extra missing')
    gp,ep=st['grid_candidates'],st['extra_candidates']
    for p,e in zip(gp+ep,grid+extra):point(p);require(p['pars'][3]==e,'Eta grid coordinate differs')
    brackets=[[grid[i-1],grid[i+1]] for i in range(1,16) if gp[i]['nll']<=gp[i-1]['nll'] and gp[i]['nll']<=gp[i+1]['nll'] and
              (gp[i]['nll']<gp[i-1]['nll'] or gp[i]['nll']<gp[i+1]['nll'])]
    runs=payload['search_records']
    require(st['local_brackets']==brackets and st['solver_runs']==len(brackets) and runs==st['runs'] and
            [r['bounds'] for r in runs]==brackets and all(r['success'] is True for r in runs) and
            payload['all_mandated_solvers_successful'] is True,'Mandatory scalar solver failure or coverage difference')
    for r in runs:
        point(r['candidate']);require(r['label']=='level_eta' and 0<r['nfev'] and r['nit']<=100 and close(r['nll'],r['candidate']['nll'],TOL),'Scalar return differs')
    require(st['eligible']==gp+ep+[r['candidate'] for r in runs] and st['selected']==min(st['eligible'],key=lambda p:p['nll']),'Scalar eligible/winner differs')
    curves=m['eta_profile'];require(len(curves)==11 and [c['eta'] for c in curves]==list(np.linspace(0.,3.,11)),'Mandatory profile missing')
    for i,c in enumerate(curves):point(c['selected']);require(c['selected']['pars'][3]==c['eta'] and c['reused_null']==(i==0),'Profile coordinate/reuse differs')
    require(len(m['null_candidates'])==2 and m['null_candidates'][0]==kk['null'] and curves[0]['selected']==m['null'],'Null floor/reuse differs')
    require(m['candidates']==[kk['alternative'],kk['null'],m['null']]+st['eligible']+[c['selected'] for c in curves],'Eligible candidate trace differs')
    for p in m['candidates']+m['null_candidates']:point(p)
    require(all(p['pars'][3]==0. for p in m['null_candidates']) and m['null']==min(m['null_candidates'],key=lambda p:p['nll']) and
            m['alternative']==min(m['candidates'],key=lambda p:p['nll']),'Null/alternative winner differs')
    require(m['complete'] is True and m['status']=='complete' and m['alternative']['nll']<=m['null']['nll']+1e-8 and
            all(m[phase]['nll']<=kk[phase]['nll']+1e-8 for phase in ('null','alternative')),'Completeness/nesting/floor failed')
    cm={r['label']:r for r in payload['old_backend_controls']};cp={r['label']:r for r in payload['cpu32_controls']}
    selected={'old_known_known_'+phase:kk[phase] for phase in ('null','alternative')}
    selected.update({'new_level_only_'+phase:m[phase] for phase in ('null','alternative')})
    corners={'level_corner_a_'+str(a)+'_eta_'+str(e):[a,d0,n,e] for a in (-12.,0.) for e in (0.,3.)}
    require(len(payload['old_backend_controls'])==len(cm)==8 and set(cm)==set(selected)|set(corners),'CUDA16 control coverage differs')
    require(len(payload['cpu32_controls'])==len(cp)==4 and set(cp)==set(selected),'CPU32 selected coverage differs')
    for label,r in cm.items():
        delta=abs(r['profile16_nll']-r['old_backend16_nll'])
        require(r['passed'] is True and delta<TOL and close(delta,r['absolute_difference'],1e-12) and
                r['pars']==(selected[label]['pars'] if label in selected else corners[label]),'Old backend control failed/detached')
        if label in selected:require(r['profile16_nll']==selected[label]['nll'],'Selected16 control detached')
        if label.startswith('old_'):
            phase='null' if label.endswith('_null') else 'alternative';saved=kk['saved_'+phase+'_nll']
            require(r['saved_primary16_nll']==saved and abs(r['old_backend16_nll']-saved)<TOL and
                    close(r['saved_absolute_difference'],abs(r['old_backend16_nll']-saved),1e-12),'Saved CUDA16 baseline reproduction failed')
    for label,r in cp.items():
        delta=abs(r['profile16_nll']-r['independent32_nll'])
        require(r['passed'] is True and delta<TOL and close(delta,r['absolute_difference'],1e-12) and
                r['profile16_nll']==selected[label]['nll'] and r['pars']==selected[label]['pars'],'CPU32 selected control failed/detached')
        if label.startswith('old_'):
            saved=kk['saved_'+r['phase']+'_nll'];require(r['saved_primary16_nll']==saved and abs(r['independent32_nll']-saved)<TOL and
                close(r['saved_absolute_difference'],abs(r['independent32_nll']-saved),1e-12),'Saved CPU32 reproduction failed')
    selected=m['selected_objectives'];require(len(selected)==2 and [r['phase'] for r in selected]==['null','alternative'],'Calibration selected phases missing')
    deltas=[]
    for r in selected:
        label='new_level_only_'+r['phase'];v16=cm[label]['old_backend16_nll'];v32=cp[label]['independent32_nll'];delta=abs(v32-v16)
        require(r['cuda16_nll']==v16 and r['cpu32_nll']==v32 and r['absolute_difference']==delta and delta<TOL and r['passed'] is True,'Calibration selected values failed/detached')
        deltas.append(delta)
    epsilon=max(1e-6,2*sum(deltas));score=payload['calibration_score'];statistic=m['contrast16']['statistic']
    require(score['statistic']==statistic and score['numerical_envelope']==epsilon and score['interval']==[statistic-epsilon,statistic+epsilon] and
            score['central_order']==16 and score['epsilon_floor']==1e-6 and score['interpretation']==SEARCH_SPEC['numerical_envelope_interpretation'],'Calibration score/envelope differs')
    for tag,item in [('old_known_known',kk),('new_level_only',m)]:
        for order,rows,field in [(16,cm,'old_backend16_nll'),(32,cp,'independent32_nll')]:
            ct=item['contrast'+str(order)];sub=2*(rows[tag+'_null'][field]-rows[tag+'_alternative'][field])
            require(close(ct['nll_subtraction_control'],sub,1e-8) and close(ct['direct_minus_subtraction'],ct['statistic']-sub,1e-8) and
                    abs(ct['direct_minus_subtraction'])<TOL and ct['statistic']>=-TOL,'Direct subtraction/nesting failed')
            require(close(ct['statistic'],2*math.fsum(ct[k] for k in ('event_difference','background_compensator_difference','trigger_compensator_difference')),1e-8),'Full compensator components differ')
            if tag.startswith('old'):require(abs(ct['statistic']-item['saved_contrast'+str(order)]['statistic'])<TOL,'Saved KK direct reproduction failed')
        require(abs(item['contrast16']['statistic']-item['contrast32']['statistic'])<TOL,'Direct order failed')
    identities=payload['common_backend_identities'];require([r['order'] for r in identities]==[16,32],'Identity coverage differs')
    for r in identities:
        o=r['order'];rows,field=(cm,'old_backend16_nll') if o==16 else (cp,'independent32_nll')
        g0,g1=(rows['old_known_known_'+phase][field]-rows['new_level_only_'+phase][field] for phase in ('null','alternative'))
        a,b=m['contrast'+str(o)],kk['contrast'+str(o)];lhs=a['statistic']-b['statistic'];rhs=2*(g1-g0)+a['direct_minus_subtraction']-b['direct_minus_subtraction']
        require(close(r['null_gain'],g0) and close(r['alternative_gain'],g1) and close(r['direct_statistic_change'],lhs) and
                close(r['direct_statistic_contraction_from_KK'],-lhs) and close(r['identity_rhs'],rhs) and close(r['residual'],lhs-rhs) and abs(r['residual'])<1e-8,'Common backend identity differs')
    require(payload['direct_controls_passed'] is True and payload['saved_baseline_controls_passed'] is True,'Direct gate flags differ')
    require(payload['sources']=={'iteration_24/code/profile_optimizer.py':PROFILE_SHA},'Pinned objective source differs')
    metrics=payload['cuda_metrics'];require(metrics['dtype']=='float64' and metrics['peak_allocated_bytes']<=2**30 and
        all(metrics[k] is False for k in ('allow_tf32','mixed_precision','fast_math_requested')),'CUDA numerical/resource policy differs')
    require(payload['feature_metadata']['clock_mode'] in ('original_projected_clock','paired_clock') and raw and payload['queries'] and
            payload['fixed_evaluations'] and not payload['evaluation_errors'],'Missing/error objective trace')
    require(payload['feature_metadata']['clock_mode']==canonical_clock(binding['clock_mode']),'Payload clock binding differs')
    for q in payload['queries']:
        p=raw[q['evaluation_id']];require(q['pars']==p['pars'] and q['nll']==p['nll'] and q['eta']==p['pars'][3],'Query detached from raw profile')
    for p in payload['fixed_evaluations']:point(dict(p,full_marked_nll=p['nll']+mark))
    return True
