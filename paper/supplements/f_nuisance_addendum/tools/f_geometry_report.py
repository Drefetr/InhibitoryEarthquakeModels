"""Saved-output arithmetic and optional plots only; no scientific evaluation."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def collect(run):
    run=Path(run);directory=run/'results/F_geometry_diagnostic06'
    protocol_path=run/'protocols/F_geometry_diagnostic06_frozen.json'
    execution_path=run/'resource/F_geometry_diagnostic06/execution.json'
    protocol=read(protocol_path)['protocol'];execution=read(execution_path)
    if not execution.get('finished_utc'):raise ValueError('ended process required')
    index=read(directory/'case_index.json');digest=sha(protocol_path)
    if index['protocol_sha256']!=digest or index['planned_ids']!=protocol['sample_plan']['case_ids']:
        raise ValueError('plan/index mismatch')
    bindings={r['id']:r for r in protocol['sample_plan']['case_bindings']}
    cases=[];solvers=[];case_pins=[]
    for case_id in index['planned_ids']:
        path=directory/(case_id+'.json');value=read(path) if path.exists() else {}
        if value and value['protocol_sha256']!=digest:raise ValueError('case receipt mismatch')
        if path.exists():case_pins.append(dict(path=str(path.relative_to(run)),sha256=sha(path)))
        diagnostic=value.get('diagnostic',{});points=diagnostic.get('points',[]);profiles=diagnostic.get('profile_rows',[])
        case=dict(id=case_id,selection_class=bindings[case_id]['selection_class'],n_events=bindings[case_id]['n_events'],
            status=value.get('status',index['statuses'][case_id]),error=value.get('error'),
            point_count=len(points),solver_rows=len(diagnostic.get('solvers',[])),planned_solver_rows=36,
            profile_rows=len(profiles),planned_profile_rows=140,order32_rows=len(value.get('order32_controls',[])),
            points=points,profiles=profiles,
            maximum_profile_agreement_error=max((r['absolute_difference'] for r in profiles),default=None),
            maximum_order32_difference=max((r['absolute_difference'] for r in value.get('order32_controls',[])),default=None))
        if profiles and points:
            supplied=points[-1]['pars'];fixed_d=[r for r in profiles if r['d']==supplied[1]]
            at0=next((r for r in fixed_d if r['eta']==0),None)
            atlaw=next((r for r in fixed_d if r['eta']==supplied[3]),None)
            if at0 is not None and atlaw is not None:
                case['fixed_d_profile']=dict(eta0_a=at0['profile']['a'],supplied_eta_a=atlaw['profile']['a'],
                    amplitude_log_shift=atlaw['profile']['a']-at0['profile']['a'],
                    supplied_eta_minus_zero_nll=atlaw['fresh_ground_objective']-at0['fresh_ground_objective'])
            best=min(r['fresh_ground_objective'] for r in profiles)
            case['mesh']=dict(minimum_ground_nll=best,within_half=sum(r['fresh_ground_objective']<=best+.5 for r in profiles),
                              within_two=sum(r['fresh_ground_objective']<=best+2 for r in profiles))
        cases.append(case)
        for row in diagnostic.get('solvers',[]):
            result=row['result'];metadata={k:v for k,v in row.items() if k!='result'}
            solvers.append(dict(case_id=case_id,selection_class=case['selection_class'],**metadata,
                **{k:v for k,v in result.items() if k not in ('trace','last_evaluation')}))
    seams=[abs(p['derivative_probe']['eta0_positive_branch_objective_seam']) for c in cases for p in c['points']
           if p['derivative_probe']['eta0_positive_branch_objective_seam'] is not None]
    return dict(classification='saved arithmetic only; not new fits or repaired outcomes',protocol_sha256=digest,
        index_sha256=sha(directory/'case_index.json'),execution_sha256=sha(execution_path),execution=execution,
        planned_cases=len(cases),status_counts=dict(collections.Counter(c['status'] for c in cases)),
        selection_counts=dict(collections.Counter(c['selection_class'] for c in cases)),
        saved_solver_rows=len(solvers),planned_solver_rows=36*len(cases),
        solver_return_success=sum(r.get('solver_success') is True for r in solvers),
        solver_return_failure=sum(r.get('solver_success') is False for r in solvers),
        solver_exceptions=sum(r['status']=='diagnostic_exception' for r in solvers),
        invalid_proposals=sum(len(r.get('invalid_proposals',[])) for r in solvers),
        saved_profile_rows=sum(c['profile_rows'] for c in cases),planned_profile_rows=140*len(cases),
        maximum_eta0_seam=max(seams,default=None),cases=cases,solvers=solvers,case_pins=case_pins)


def plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
    image=None
    for ax,case in zip(axes.flat,summary['cases']):
        ax.set_title(case['id']+'\n'+case['status'],fontsize=9)
        rows=case['profiles']
        if not rows:
            ax.text(.5,.5,'No profile mesh saved',ha='center',va='center',transform=ax.transAxes);ax.set_axis_off();continue
        d=sorted({r['d'] for r in rows});eta=sorted({r['eta'] for r in rows});best=min(r['fresh_ground_objective'] for r in rows)
        z=np.full((len(d),len(eta)),np.nan)
        for r in rows:z[d.index(r['d']),eta.index(r['eta'])]=r['fresh_ground_objective']-best
        image=ax.pcolormesh(eta,d,z,shading='nearest',vmin=0,vmax=20,cmap='viridis')
        ax.set_xlabel('eta');ax.set_ylabel('d');ax.set_xlim(0,3);ax.set_ylim(-2,2)
    for ax in list(axes.flat)[len(summary['cases']):]:ax.set_axis_off()
    if image is not None:fig.colorbar(image,ax=list(axes.flat),label='Profile ground NLL minus saved mesh minimum (colour saturates at 20)')
    fig.suptitle('All 11 planned cases; conditional amplitude profiles, not confidence regions')
    fig.savefig(output/'profile_meshes.png',dpi=160);plt.close(fig)
    fig,(left,right)=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    positive_error_count=0;zero_errors=[]
    for index,case in enumerate(summary['cases']):
        old=case['selection_class'].startswith('old_failure');marker='x' if old else 'o'
        color='C'+str(index%10)
        available=[p for p in case['points'] if p['realized_event_geometry']['eta_defined']]
        if available:left.scatter([p['pars'][3] for p in available],
            [p['realized_event_geometry']['eta_residual_fraction'] for p in available],marker=marker,color=color,label=case['id'])
        point=next((p for p in case['points'] if p['pars'][3]==0),None)
        if point:
            rows=sorted((r for r in point['derivative_probe']['rows'] if r['coordinate']==3),key=lambda r:r['step'])
            errors=np.asarray([r['absolute_error'] for r in rows],dtype=float)
            if not np.isfinite(errors).all() or np.any(errors<0):raise ValueError('invalid saved absolute error')
            positive_error_count+=int(np.count_nonzero(errors>0))
            zeros=[r['step'] for r in rows if r['absolute_error']==0]
            if zeros:zero_errors.append(case['id']+': '+', '.join(format(h,'.0e') for h in zeros))
            # Display mask only: preserve raw errors and break lines at zero.
            if np.any(errors>0):right.plot([r['step'] for r in rows],np.ma.masked_equal(errors,0),marker=marker,color=color,label=case['id'])
    left.set(xlabel='Fixed eta',ylabel='Eta residual fraction after a,d projection',title='Realized-event geometry; x = selected old failure witness')
    right.set(xlabel='One-sided eta=0 difference step',ylabel='Absolute finite-difference derivative error',title='All four saved steps; no best-step selection')
    right.set_xscale('log')
    if positive_error_count:
        right.set_yscale('log');right.legend(fontsize=7)
    else:
        right.set_axis_off();right.text(.5,.5,'No positive errors; log display not applicable',ha='center',transform=right.transAxes)
    zero_note='Zero errors: none' if not zero_errors else 'Zeros not placed on log axis (case: step):\n'+'\n'.join(zero_errors)
    right.text(.02,.02,zero_note,va='bottom',fontsize=7,transform=right.transAxes,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    left.legend(fontsize=7)
    fig.savefig(output/'geometry_and_derivative_steps.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--plot',action='store_true');args=parser.parse_args()
    summary=collect(args.run);args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    fields=['id','selection_class','n_events','status','point_count','solver_rows','planned_solver_rows','profile_rows','planned_profile_rows','order32_rows']
    with (args.output/'case_accounting.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:c[k] for k in fields} for c in summary['cases'])
    if args.plot:plots(summary,args.output)
    print(json.dumps({k:summary[k] for k in ('planned_cases','status_counts','saved_solver_rows','planned_solver_rows','saved_profile_rows','planned_profile_rows','maximum_eta0_seam')}))


if __name__=='__main__':main()
