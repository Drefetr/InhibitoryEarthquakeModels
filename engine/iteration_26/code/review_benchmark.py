"""Bounded benchmark controls and prospective resource gate; never calibrates scores."""
from pathlib import Path
import argparse
import csv
import io
import json
import math
import statistics
import sys
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import read_sealed,sha256,sealed_json,canonical_hash,atomic_bytes,utc_now
from run_level_calibration import AREA,BENCHMARK_IDS,PILOT_IDS,IDS,guard,cached,require,lifetime


def collect():
    began=time.perf_counter();run=AREA/'results/benchmark'
    with __import__('checkpoint_io').RunLock(run/'pilot.lock'):
        plan=read_sealed(run/'plan.json');summary=read_sealed(run/'summary.json');ledger=read_sealed(run/'sessions.json')
        guard(plan)
        require(plan['contract']['stage']=='benchmark' and plan['ids']==BENCHMARK_IDS and summary['states']=={i:'complete' for i in BENCHMARK_IDS} and
                summary['all_complete'] and summary['terminal_complete'] and summary['sessions_closed'] and not summary['active'] and not summary['active_ids'] and
                ledger['sessions'] and all(s['closed'] and s['stage']=='closed' and not s.get('active_id') for s in ledger['sessions']), 'Closed complete12 benchmark required')
        sessions=ledger['sessions'];charge=sum(s['elapsed_seconds'] for s in sessions)
        require(summary['plan_id']==ledger['plan_id']==plan['plan_id'] and summary['stage']=='benchmark' and summary['total_planned']==12 and
                summary['counts']=={'complete':12} and len({s['session_id'] for s in sessions})==len(sessions),'Benchmark terminal lineage/quota differs')
        require(not (run/'fatal.json').exists() and all(not s.get('active_attempt_id') and not s.get('error') and not s.get('worker_error') and
                s.get('worker_exitcode',0)==0 and math.isfinite(s['elapsed_seconds']) and s['elapsed_seconds']>=0 for s in sessions) and
                charge==summary['charged_seconds'] and charge<=2400,'Unreviewed failure or invalid benchmark accounting')
        for session in sessions:lifetime().ensure_old_worker_dead(session)
        evidence={};rows=[];by={e['id']:e for e in plan['entries']}
        for name in ('plan.json','summary.json','sessions.json'):evidence[(run/name).relative_to(ROOT).as_posix()]=sha256(run/name)
        for i in BENCHMARK_IDS:
            case=cached(plan,by[i],run);require(case is not None and case['status']=='complete','Unresolved benchmark case')
            path=run/'cases'/(i+'.json');evidence[path.relative_to(ROOT).as_posix()]=sha256(path)
            payload=case['payload'];m=payload['methods'][0]
            oldpath=ROOT/plan['benchmarks'][i]['path'];old=read_sealed(oldpath)
            row=dict(id=i,n_events=by[i]['n_events'],case_seconds=case['elapsed_seconds'],
                     preparation32_seconds=payload['feature_metadata']['preparation32_seconds'],
                     old21_case_seconds=old['elapsed_seconds'],old21_preparation32_seconds=old['payload']['methods'][0]['cpu32_feature_metadata']['preparation_seconds'],
                     max_overlap25_selected_nll_difference=None)
            if i in PILOT_IDS:
                old25path=ROOT/f'iteration_25/results/ablation_pilot/cases/{i}.json';old25=read_sealed(old25path)
                previous=next(m for m in old25['payload']['methods'] if m['id']=='level_only')
                row['max_overlap25_selected_nll_difference']=max(abs(m[p]['nll']-previous[p]['nll']) for p in ('null','alternative'))
                evidence[old25path.relative_to(ROOT).as_posix()]=sha256(old25path)
            rows.append(row)
        full=read_sealed(AREA/'results/full_calibration/plan.json')
        # Count-stratified residual estimate plus full historical CPU32 feature timings.
        old_total_feature=0.;old_total_elapsed=0.
        for i in IDS:
            oldpath=ROOT/f'iteration_21/results/full_staged/cases/{i}.json'
            require(sha256(oldpath)==full['benchmarks'][i]['sha256'],'Uniform old baseline changed')
            old=read_sealed(oldpath);old_total_elapsed+=old['elapsed_seconds']
            old_total_feature+=old['payload']['methods'][0]['cpu32_feature_metadata']['preparation_seconds']
        ratio=statistics.median(r['preparation32_seconds']/r['old21_preparation32_seconds'] for r in rows)
        null_res=statistics.median(max(0.,r['case_seconds']-r['preparation32_seconds']) for r in rows if r['id'].startswith('null_') and r['id']!='null_1292')
        alt_res=statistics.median(max(0.,r['case_seconds']-r['preparation32_seconds']) for r in rows if r['id'].startswith('new_'))
        extreme=next(r for r in rows if r['id']=='null_1292')
        stratified=old_total_feature*ratio+1998*null_res+150*alt_res+max(0.,extreme['case_seconds']-extreme['preparation32_seconds'])
        historical_ratio=sum(r['case_seconds'] for r in rows)/sum(r['old21_case_seconds'] for r in rows)
        paired_forecast=old_total_elapsed*historical_ratio
        # Prospective heuristic safety multiplier; neither estimate is a statistical interval.
        conservative=1.5*max(stratified,paired_forecast)+60+15+1
        limits=full['contract']['resources'];overlap=max(r['max_overlap25_selected_nll_difference'] or 0. for r in rows)
        passed=overlap<1e-4 and conservative<=limits['total_parent_wall_seconds']
        out=AREA/'results/benchmark_review';require(not out.exists(),'Immutable benchmark review already present');out.mkdir()
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        atomic_bytes(out/'costs.csv',output.getvalue().encode())
        result=dict(created_utc=utc_now(),benchmark_plan_id=plan['plan_id'],full_plan_id=full['plan_id'],source_hashes=plan['sources'],
                    cases_complete=12,charged_parent_seconds=summary['charged_seconds'],all_numerical_gates_passed=True,
                    overlap25_max_selected_nll_difference=overlap,uniform21_seed_change_passed=overlap<1e-4,
                    forecast=dict(stratified_seconds=stratified,paired_historical_ratio_seconds=paired_forecast,
                                  heuristic_safety_seconds=conservative,historical_cpu32_feature_total_seconds=old_total_feature,
                                  historical_case_total_seconds=old_total_elapsed,feature_ratio_median=ratio,
                                  null_residual_median_seconds=null_res,alternative_residual_median_seconds=alt_res,
                                  limitations='Selected12 cost sample; two heuristic forecasts, not confidence bounds. No calibration or power computed.'),
                    passed=passed,evidence_hashes=evidence,elapsed_review_seconds=time.perf_counter()-began)
        sealed_json(out/'summary.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','evidence_hashes')},indent=2))
        return 0 if passed else 2


if __name__=='__main__':raise SystemExit(collect())
