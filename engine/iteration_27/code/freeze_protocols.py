"""Preparation only: pin reviewed-source candidates before any scientific outcome."""
from pathlib import Path
import json,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import sha256,sealed_json,utc_now

def main():
    dest=ROOT/'iteration_27/configs'
    if any((dest/p).exists() for p in ['observed_frozen.json','horizon_frozen.json']):raise FileExistsError('Immutable protocol exists')
    interface=json.loads((dest/'D_interface.json').read_text())
    old=json.loads((ROOT/'iteration_26/results/full_calibration/plan.json').read_text())
    paths=list(old['sources'])+['iteration_12/code/clock_simulator.py']
    paths+=['iteration_27/code/'+s for s in ['adequacy.py','run_adequacy.py','horizon_pilot.py','bounded_run.py','analyze_adequacy.py','freeze_protocols.py']]
    paths+=['iteration_27/configs/D_interface.json','iteration_27/tests/test_adequacy.py','iteration_27/tests/test_horizon_pilot.py','iteration_27/tests/test_analysis.py']
    sources={p:sha256(ROOT/p) for p in sorted(set(paths))}
    historical={p:sha256(ROOT/p) for p in ['iteration_5/configs/empirical_protocol.json','iteration_5/reviews/C_reconstructed_prelaunch_protocol.json','iteration_5/reviews/C_protocol_chronology.json','iteration_5/results/B_primary_empirical.json']}
    for p,h in old['sources'].items():
        if sources[p]!=h:raise ValueError('Prior authority changed '+p)
    observed=dict(schema_version=1,status='frozen_before_new_observed_outcomes',created_utc=utc_now(),
        objective='Exploratory temporal model adequacy on previously seen observed GeoNet reconstruction; no physical mechanism or power inference.',
        T0=24394.,wall_cap_seconds=1800,source_hashes=sources,
        historical_authorities=historical,
        data=dict(path='iteration_5/data/processed/geonet_M5_shallow_1960_2026.csv',sha256='feacccb7581d1c4c33f88874cd6b198c41639da6930a7fc35e46f910a594fb3e',start='1960-01-01',end_exclusive='2026-10-15',n=913),
        folds=[dict(id='fold_50',train_fraction=.5,test_end_fraction=.65),dict(id='fold_65',train_fraction=.65,test_end_fraction=.8),dict(id='fold_80',train_fraction=.8,test_end_fraction=1.)],
        search=interface['search'],interface=interface,
        training='Strict prefix, beta from training marks alone and common across families, original absolute x=2t/T0-1; finite search and bounded local solvers, no global optimality claim.',
        scoring='Full predictable conditional ground and common mark loglikelihood, all earlier history, random count, full compensator and terminal censored exposure.',
        primary_comparisons=['ml_shared minus ml_trigger','etas_omori minus ml_trigger','etas_omori minus ml_shared','ml_trigger minus poisson_loglinear'],
        descriptive_diagnostics=['Uniform KS statistic without p-value','centered ACF lags1 through10','mean complete compensator gap','count minus realized compensator','residual ECDF and compensator curve'],
        failure_policy='Every prescribed slot must have accepted successful lineage; retain all failed raw attempts. Stop on first unresolved search/control/resource issue. No partial estimator or post-outcome grid reduction.',
        effect_scope='Predictive preference is not physical identification; all histories/families previously inspected, exploratory only.',
        execution=dict(script='iteration_27/code/run_adequacy.py',output='iteration_27/results/observed.json',receipt='iteration_27/logs/observed_execution.json',outer_seconds=1800,native_threads=1,closure_operational_allowance_seconds=1,proved_bound=False),
        postanalysis=dict(script='iteration_27/code/analyze_adequacy.py',output='iteration_27/results/observed_analysis',wall_cap_seconds=60,separate_from_refit_cap=True),
        historical_protocol_qualification='Current iteration5 protocol differs from old execution hash. Separately preserved reconstructed version matches, but was not a contemporaneous immutable launch snapshot. This new protocol is independently frozen.')
    law=old['contract']['law'];fit=law['alternative_generating_fit']
    seeds={arm:[int(np.random.SeedSequence([20261018,27,rep,index]).generate_state(1,dtype=np.uint64)[0]) for rep in range(5)] for index,arm in enumerate(['null','inhibited'])}
    ids=[f'{arm}_r{rep:02d}_h{mult}' for rep in range(5) for mult in [1,2,4] for arm in ['null','inhibited']]
    horizon=dict(schema_version=1,status='frozen_before_horizon_pilot_outcomes',created_utc=utc_now(),wall_cap_seconds=600,source_hashes=sources,
        generating_law=dict(T0=24394.,beta=law['beta'],eta=law['alternative_eta'],fit=fit),seeds=seeds,planned_ids=ids,
        max_events=20000,max_proposals=2000000,simulation_seconds=20,fit_seconds=120,validation_reserve_seconds=30,cleanup_reserve_seconds=5,maxls_tiers=[40,200],maxiter=200,
        physical_background='Fixed original exp(a0+d0*(2t/T0-1)); transform simulator coordinates aH=a0+d0*(H/T0-1),dH=d0*H/T0. Not stationary extrapolation or fixed-rate count multiplier.',
        inference_scope='Feasibility only, no5percent calibration, critical values, ranks, rejection proportions or power estimate. Five independent long paths per law, nested dependent prefixes.',
        fitting='Supplied beta, productivity ratio and shared195-mode kernel; free a,d,n null and a,d,n,eta alternative; two generic starts each with own-null floor; all4slots must succeed. Not full-nuisance empirical recipe.',
        numerical_gates='Selected GPU16 vs fresh CPU32 NLL and direct fullcompensator contrasts each<1e-4; exact Decimal exclusive prefix and precision-aware paired clocks; retain extremes.',
        failure_policy='Stop first unresolved simulation/fit/control/resource issue; retain censored path+state, completed fits and every planned pending case; no replacement seeds or truncation as success.',
        ordering='Rep outer, simulate both long arms, horizon1/2/4 middle and null/inhibited inner.',
        execution=dict(script='iteration_27/code/horizon_pilot.py',output='iteration_27/results/horizon_pilot',receipt='iteration_27/logs/horizon_execution.json',outer_seconds=600,closure_operational_allowance_seconds=1,proved_bound=False),
        future_campaign='Prepare configuration/cost only, no automatic new null bank,500pairs,nestedbootstrap,catalogue download or trend-only campaign.')
    for name,p in [('observed_frozen.json',observed),('horizon_frozen.json',horizon)]:sealed_json(dest/name,p)
    print(json.dumps({f'iteration_27/configs/{name}':sha256(dest/name) for name in ['observed_frozen.json','horizon_frozen.json']},indent=2))

if __name__=='__main__':main()
