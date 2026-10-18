"""Bounded observed rolling diagnostics; --benchmark uses synthetic data only.

Outputs are append-only, with atomic progress checkpoints. Real refits require
an explicitly hash-bound frozen protocol. No simulation calibration is run.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
from threadpoolctl import threadpool_limits
from adequacy import (ROOT, BudgetExceeded, check_deadline, fit_training,
                      selected_controls, validate_fit)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_atomic(path, payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def validate_protocol(protocol, interface, benchmark=False):
    if protocol['search']!=interface['search']:raise ValueError('search specification mismatch')
    if protocol['T0']!=24394.:raise ValueError('physical time basis must remain original T0')
    expected=[{'id':'fold_50','train_fraction':.5,'test_end_fraction':.65},
              {'id':'fold_65','train_fraction':.65,'test_end_fraction':.8},
              {'id':'fold_80','train_fraction':.8,'test_end_fraction':1.}]
    if protocol['folds']!=expected:raise ValueError('rolling window specification mismatch')
    if not 0<protocol['wall_cap_seconds']<=1800:raise ValueError('invalid declared wall cap')
    if not benchmark and protocol.get('status')!='frozen_before_new_observed_outcomes':
        raise ValueError('observed fits require frozen protocol')
    for rel, expected_sha in protocol.get('source_hashes',{}).items():
        if sha(ROOT/rel)!=expected_sha:raise ValueError(f'source binding changed: {rel}')
    required=['iteration_27/code/adequacy.py','iteration_27/code/run_adequacy.py',
              'iteration_27/configs/D_interface.json','iteration_5/code/kernels.py']
    if not benchmark and any(p not in protocol.get('source_hashes',{}) for p in required):
        raise ValueError('missing runtime source binding')


def run(protocol, protocol_path, protocol_sha, output, benchmark=False, synthetic_events=913):
    began=time.perf_counter();deadline=began+protocol['wall_cap_seconds']
    if Path(output).exists():raise FileExistsError('refusing to replace existing output')
    interface_path=ROOT/'iteration_27/configs/D_interface.json'
    interface=json.loads(interface_path.read_text(encoding='utf-8'))
    validate_protocol(protocol,interface,benchmark)
    if sha(protocol_path)!=protocol_sha:raise ValueError('protocol byte hash mismatch')
    if json.loads(Path(protocol_path).read_text(encoding='utf-8'))!=protocol:raise ValueError('protocol object/bytes mismatch')
    if benchmark:
        rng=np.random.default_rng(20261018027)
        times=np.sort(rng.uniform(0.,protocol['T0'],synthetic_events))
        excess=rng.exponential(1/2.8,synthetic_events)
        input_binding={'synthetic_only':True,'seed':20261018027,'n_events':synthetic_events}
    else:
        path=ROOT/protocol['data']['path']
        if sha(path)!=protocol['data']['sha256']:raise ValueError('catalogue byte hash mismatch')
        frame=pd.read_csv(path);times=frame.t_days.to_numpy();excess=frame.excess.to_numpy()
        input_binding={'synthetic_only':False,'path':protocol['data']['path'],'sha256':sha(path),
                       'n_events':len(times),'all_data_previously_seen':True}
    array_binding={k:hashlib.sha256(np.ascontiguousarray(v,dtype='<f8').tobytes()).hexdigest()
                   for k,v in [('times_f64le',times),('excess_f64le',excess)]}
    out={'schema_version':1,'complete':False,'status':'running','synthetic_only':benchmark,
         'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_sha,
         'input_binding':input_binding,'array_binding':array_binding,'T0':protocol['T0'],
         'wall_cap_seconds':protocol['wall_cap_seconds'],'folds':{},
         'source_hashes':{p:sha(ROOT/p) for p in ['iteration_27/code/adequacy.py',
             'iteration_27/code/run_adequacy.py','iteration_27/configs/D_interface.json','iteration_5/code/kernels.py']},
         'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                        'pandas':pd.__version__,'native_threads':1},
         'interpretation':'Previously seen observed data; exploratory expanding training windows and realized-history predictive diagnostics. No fitted-KS, Poisson-count, or mechanism p-values.'}
    write_atomic(output,out)
    try:
        with threadpool_limits(limits=1):
            for spec in protocol['folds']:
                check_deadline(deadline)
                fold_started=time.perf_counter();train_end=spec['train_fraction']*protocol['T0'];test_end=spec['test_end_fraction']*protocol['T0']
                fold={'spec':spec,'train_end':train_end,'test_end':test_end,'status':'fitting',
                      'completed_optimization_records':[]}
                out['folds'][spec['id']]=fold
                def progress(rec):
                    fold['completed_optimization_records'].append(rec)
                    # Completion-based snapshots only, preserving all search
                    # rows in the finished fold, with no periodic timer writes.
                    out['elapsed_seconds_before_checkpoint']=time.perf_counter()-began
                    write_atomic(output,out)
                fit=fit_training(times,excess,train_end,protocol['T0'],protocol['search'],deadline,progress)
                fold['fit']=fit;fold.pop('completed_optimization_records',None)
                fold['status']='validating' if fit['complete'] else 'incomplete_search'
                write_atomic(output,out)
                if not fit['complete']:raise BudgetExceeded('incomplete or failed mandatory training search; stop for review')
                validate_fit(fit,protocol['search'])
                fold['training_controls']={};fold['test_controls']={}
                for family,record in fit['best'].items():
                    check_deadline(deadline)
                    fold['training_controls'][family]=selected_controls(times,excess,record,0.,train_end,protocol['T0'],fit['beta_train'])
                    check_deadline(deadline)
                    fold['test_controls'][family]=selected_controls(times,excess,record,train_end,test_end,protocol['T0'],fit['beta_train'])
                    check_deadline(deadline)
                    write_atomic(output,out)
                passed=all(c['pass'] for key in ('training_controls','test_controls') for c in fold[key].values())
                fold['complete']=passed;fold['status']='complete' if passed else 'numerical_control_failed'
                fold['elapsed_seconds']=time.perf_counter()-fold_started
                write_atomic(output,out)
                if not passed:raise ValueError('selected numerical controls failed; stop for review')
        check_deadline(deadline)
        out['complete']=True;out['status']='complete'
    except (BudgetExceeded,ValueError,MemoryError,FloatingPointError) as exc:
        out['status']='incomplete';out['error']=f'{type(exc).__name__}: {exc}'
    out['elapsed_seconds_before_final_publication']=time.perf_counter()-began
    out['completed_utc']=datetime.now(timezone.utc).isoformat()
    out['accounting_scope']='Parent wall through final payload assembly; final atomic serialization/write and process teardown excluded. Deadline is cooperative; this is not GPU time or a proved hard cap.'
    write_atomic(output,out)
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--synthetic-events',type=int,default=913)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('refusing to replace existing output')
    if args.synthetic_events<8:raise ValueError('benchmark requires >=8 synthetic events')
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    out=run(protocol,args.protocol,args.protocol_sha256,args.output,args.benchmark,args.synthetic_events)
    print(json.dumps({'status':out['status'],'complete':out['complete'],'synthetic_only':out['synthetic_only'],
                      'elapsed_seconds_before_final_publication':out['elapsed_seconds_before_final_publication']}),flush=True)
    return 0 if out['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
