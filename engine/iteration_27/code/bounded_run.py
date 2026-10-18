"""Owned Windows child execution with whole-invocation wall cap and durable receipts."""
from pathlib import Path
import argparse,contextlib,importlib.util,math,multiprocessing as mp,os,runpy,sys,time,traceback
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'iteration_6/code'))
from checkpoint_io import ChildJob,RunLock,read_sealed,sealed_json,sha256,utc_now


def life():
    p=ROOT/'iteration_19/code/run_full_cuda.py'
    if sha256(p)!='93f59620d6f7e9a0eccbf1ae3575429634080459b1ea38601b070a89ee906dbb':raise ValueError('Lifetime authority changed')
    spec=importlib.util.spec_from_file_location('_bounded27_life',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def child(ready,script,arguments,log):
    sys.dont_write_bytecode=True
    if not ready.wait(15):raise SystemExit(2)
    with open(log,'x',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        try:
            sys.path.insert(0,str(script.parent))
            sys.argv=[str(script)]+arguments;runpy.run_path(str(script),run_name='__main__')
        except SystemExit:raise
        except BaseException:traceback.print_exc();raise SystemExit(2)


def execute(script,arguments,seconds,receipt,gate=None):
    began=time.perf_counter();receipt=Path(receipt);receipt.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists() or receipt.with_suffix('.started.json').exists():raise FileExistsError('Existing bounded generation; choose new declared output')
    if os.name!='nt':raise RuntimeError('Windows job containment required')
    scientific={'iteration_27/code/run_adequacy.py','iteration_27/code/horizon_pilot.py'}
    if script.relative_to(ROOT).as_posix() in scientific and gate is None:raise ValueError('Scientific execution requires actual A/C gate')
    if gate is not None:
        g=read_sealed(gate)
        if g['passed'] is not True:raise ValueError('Unapproved execution gate')
        if len(arguments)!=6 or set(arguments[::2])!={'--protocol','--protocol-sha256','--output'}:raise ValueError('Only exactly three unique scientific flag/value pairs allowed')
        protocol=Path(arguments[arguments.index('--protocol')+1]).resolve()
        actual=sha256(protocol)
        if g['protocol_sha256'].get(protocol.relative_to(ROOT).as_posix())!=actual:raise ValueError('Gate protocol differs')
        if arguments[arguments.index('--protocol-sha256')+1]!=actual:raise ValueError('CLI protocol hash differs')
        if script.relative_to(ROOT).as_posix() not in g['approved_scripts']:raise ValueError('Gate script differs')
        spec=g['approved_invocations'][protocol.relative_to(ROOT).as_posix()]
        if not math.isfinite(seconds) or seconds!=spec['seconds'] or seconds!=read_sealed(protocol)['wall_cap_seconds']:raise ValueError('Outer cap differs from frozen cap')
        if spec['script']!=script.relative_to(ROOT).as_posix():raise ValueError('Script/protocol pair differs')
        if (ROOT/spec['receipt']).resolve()!=receipt.resolve():raise ValueError('Receipt differs')
        if (ROOT/spec['output']).resolve()!=Path(arguments[arguments.index('--output')+1]).resolve():raise ValueError('Scientific output differs')
        for path,h in g['source_hashes'].items():
            if sha256(ROOT/path)!=h:raise ValueError('Gated source changed: '+path)
        for role,path in g['review_roles'].items():
            review=read_sealed(ROOT/path)
            if sha256(ROOT/path)!=g['review_sha256'][path] or review['role']!=role or review['passed'] is not True or review['protocol_sha256']!=g['protocol_sha256']:raise ValueError('Actual review binding differs')
        if set(g['review_roles'])!={'A','C'}:raise ValueError('A/C reviews required')
    with RunLock(receipt.with_suffix('.lock')):
        api=life();ctx=mp.get_context('spawn');ready=ctx.Event();log=receipt.with_suffix('.console.log')
        process=ctx.Process(target=child,args=(ready,script,arguments,log));job=ChildJob();identity=None;reason='error'
        try:
            process.start();job.assign(process);identity=api.process_identity(process.pid)
            if identity is None:raise RuntimeError('Owned creation token unavailable')
            sealed_json(receipt.with_suffix('.started.json'),dict(started_utc=utc_now(),script=str(script),script_sha256=sha256(script),arguments=arguments,cap_seconds=seconds,worker_identity=identity,initialization_seconds=time.perf_counter()-began))
            ready.set();deadline=began+seconds-5
            while process.is_alive() and time.perf_counter()<deadline:process.join(.2)
            reason='complete' if not process.is_alive() and process.exitcode==0 else 'child_nonzero' if not process.is_alive() else 'outer_wall_cap'
        except KeyboardInterrupt:reason='interrupted'
        finally:
            if process.pid is not None:
                if process.is_alive():process.terminate()
                process.join(2)
            job.close()
            if process.pid is not None:process.join(2)
            alive=identity is not None and api.process_identity(identity['pid'])==identity
            elapsed=time.perf_counter()-began
            sealed_json(receipt,dict(completed_utc=utc_now(),reason=reason,exit_status=process.exitcode,worker_identity=identity,matching_generation_alive=alive,
                        measured_elapsed_seconds_before_receipt=elapsed,charged_seconds=elapsed+1,closure_operational_allowance_seconds=1,
                        closure_allowance_is_proven_bound=False,cap_seconds=seconds,script_sha256=sha256(script),arguments=arguments,console_path=str(log),
                        complete=reason=='complete' and not alive and elapsed+1<=seconds))
        return 0 if reason=='complete' and not alive and elapsed+1<=seconds else 2


if __name__=='__main__':
    mp.freeze_support();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,required=True);p.add_argument('--receipt',type=Path,required=True);p.add_argument('--script',type=Path,required=True);p.add_argument('--gate',type=Path);p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.seconds<=6:raise ValueError('At least6seconds required for cleanup reservation')
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    raise SystemExit(execute(a.script.resolve(),args,a.seconds,a.receipt.resolve(),a.gate))
