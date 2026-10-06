"""Targeted R7 mechanisms and frozen, episode-timed paired comparisons.

Reuse R6's hardened physical factories, full-system evaluator, and work ledger.
No historical batch reruns. Received data/reference calculation and persistence
are outside the timed deployment episode. Every repetition resets all state.
"""
import argparse
from dataclasses import replace, asdict
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.linalg import eigvalsh
import research_v6 as r6
from waveforge6g.receivers.operator_reuse import (ValidatorPolicy, OperatorReuseValidator,
    ValidatedRefiner, PayloadReuseRefiner, ScalarEnvelopePreparation, relative_scale,
    aligned_change, sparse_norm_upper)
from waveforge6g.receivers.certiphy_reuse import ReuseState,ReusePolicy,ReuseCertiPHY,channel_norm
from waveforge6g.receivers.certiphy import StopRule,WorkLedger
from waveforge6g.receivers.certiphy_deep import circulant_envelope
from waveforge6g.core.modulation import bits_per_symbol,modulate,demodulate
from waveforge6g.channels.reuse_paths import path_change_bound

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research_v7';DOC=ROOT/'docs/research_v7';CFG=ROOT/'configs/research_v7'
for p in [OUT,DOC,CFG]:p.mkdir(parents=True,exist_ok=True)
METHODS=['gray','radau','r6_deep','merged_absolute','phase_absolute','phase_relative','relative_radau','residual','stable']


def write(p,x):
    r6.write(p,x)


def sources():
    _,names=r6.sources()
    names += ['src/waveforge6g/receivers/payload_selection.py','src/waveforge6g/receivers/operator_reuse.py','scripts/research_v7.py']
    h=hashlib.sha256()
    for name in names:h.update(name.encode());h.update((ROOT/name).read_bytes())
    return h.hexdigest(),names


class Budget:
    def __init__(self):
        self.old=json.loads((OUT/'budget.json').read_text()) if (OUT/'budget.json').exists() else dict(seconds=0,receiver_calls=0)
        self.started=time.perf_counter();self.calls=0
    def check(self):
        if self.old['seconds']+time.perf_counter()-self.started>=14400:raise RuntimeError('R7 four-hour cap; saved episodes retained')
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>5*1024**3:raise RuntimeError('R7 5 GiB cap')
    def save(self):
        write(OUT/'budget.json',dict(seconds=self.old['seconds']+time.perf_counter()-self.started,
              receiver_calls=self.old['receiver_calls']+self.calls,
              bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())))


def prepare_data(case,seed,name,epochs):
    w=r6.wave(case,name);nu=10**(-case['snr']/10);m=case['payload_symbols'];width=bits_per_symbol(case['modulation'])
    frames=[];ys=[];truths=[];refs=[];knowns=[];radii=[]
    for epoch in epochs:
        c=r6.frame(case,seed,epoch);rng=np.random.default_rng(seed*1000+epoch)
        truth=rng.integers(0,2,m*width,dtype=np.uint8)
        s=np.r_[modulate(truth,case['modulation']),np.zeros(case['n']-m,complex)]
        y=c.apply(w.modulate(s))[case['cp']:]+(rng.normal(size=case['n'])+1j*rng.normal(size=case['n']))*np.sqrt(nu/2)
        # The dense reference is evaluator-only and is never timed as deployment.
        evaluator=ReuseCertiPHY(w,c,nu,case['modulation'],ReuseState(ReusePolicy(diagonal_age=1)))
        soft,known,radius=r6.reference(evaluator,y)
        frames.append(c);ys.append(y);truths.append(truth)
        refs.append(demodulate(soft[:m],case['modulation']));knowns.append(known[:m*width]);radii.append(radius)
    return frames,np.array(ys),np.array(truths),np.array(refs),np.array(knowns),np.array(radii)


def specification(method,tolerance=.003,stable_checks=4,delta=.01):
    spec=dict(kind='gray',deep=False,validator=False,alignment='row',relative=True,
              tolerance=tolerance,stable_checks=stable_checks,delta=delta,full=False)
    if method=='r6_deep':spec['deep']='r6'
    elif method in ['merged_absolute','phase_absolute','phase_relative','common_relative']:
        spec['deep']='new';spec['alignment']='none' if method=='merged_absolute' else 'common' if method=='common_relative' else 'row'
        spec['relative']=method in ['phase_relative','common_relative']
    elif method=='relative_radau':spec['kind']='radau';spec['validator']=True
    elif method in ['radau','residual','stable']:
        spec['kind']={'radau':'radau','residual':'residual_fast','stable':'stable_fast'}[method]
    elif method=='full_r6_deep':spec.update(deep='r6',full=True)
    elif method.startswith('residual_'):spec.update(kind='residual_fast',tolerance=float(method.split('_',1)[1]))
    elif method.startswith('stable_'):spec.update(kind='stable_fast',stable_checks=int(method.split('_',1)[1]))
    elif method!='gray':raise ValueError(method)
    return spec


def deploy_episode(case,name,frames,ys,spec,budget,keep_trace=False):
    # Equal preallocated return buffers. Serialization and evaluator metrics follow
    # this entire batch timer, with no other receiver family in the timed block.
    results=[None]*len(frames)
    transitions=[None]*len(frames);storage=[0]*len(frames)
    start,cpu=time.perf_counter_ns(),time.process_time_ns()
    w=r6.wave(case,name);st=ReuseState(ReusePolicy(minimum_scale=.25,diagonal_age=1))
    nu=10**(-case['snr']/10);m=case['payload_symbols']
    for i,(c,y) in enumerate(zip(frames,ys)):
        rx=ReuseCertiPHY(w,c,nu,case['modulation'],st)
        refiner=preparer=validator=None
        if spec['deep']=='r6':refiner=PayloadReuseRefiner(st)
        elif spec['deep']=='new' or spec['validator']:
            validator=OperatorReuseValidator(st,ValidatorPolicy(spec['alignment'],spec['relative'],.25,10),directions=not spec['validator'])
            if spec['deep']=='new':refiner=ValidatedRefiner(validator)
            else:preparer=ScalarEnvelopePreparation(validator)
        requested=spec['delta']*(m/case['n'] if spec['full'] else 1.)
        rule=StopRule(method=spec['kind'],delta=requested,max_iterations=256,
                      period=2 if spec['kind'].endswith('_fast') else 4,
                      schedule='periodic' if spec['kind'].endswith('_fast') or keep_trace else 'adaptive',
                      tolerance=spec['tolerance'],stable_checks=spec['stable_checks'])
        r=rx.solve(y,rule,payload_indices=None if spec['full'] else np.arange(m),
                   refiner=refiner,preparer=preparer,keep_trace=keep_trace)
        # These compact diagnostics are returned by the implementation; no JSON,
        # reference answer, BER calculation or variable result collector is timed.
        transitions[i]={} if refiner is None and validator is None else refiner.transition if refiner is not None else validator.transition
        storage[i]=0 if not hasattr(st,'anchor_matrix') else int(st.anchor_matrix.data.nbytes+st.anchor_matrix.indices.nbytes+st.anchor_matrix.indptr.nbytes+st.anchor_rows.nbytes)
        results[i]=r
    cpu_s=(time.process_time_ns()-cpu)*1e-9;elapsed=(time.perf_counter_ns()-start)*1e-9
    for r,t,memory in zip(results,transitions,storage):
        r['transition']=t;r['anchor_storage']=memory
    budget.calls+=len(frames)
    return results,dict(cpu_s=cpu_s,elapsed_s=elapsed)


def records(case,seed,name,epochs,method,results,refs,knowns,truth,spec):
    b=case['payload_symbols']*bits_per_symbol(case['modulation']);rows=[]
    for epoch,r,ref,known,true in zip(epochs,results,refs,knowns,truth):
        output=r['bits'][:b];t=r['transition'];diff=(output!=ref)
        conservative=float(np.mean(diff|(~known)))
        bound=r['disagreement_bound']*(case['n']/case['payload_symbols'] if spec['full'] else 1.)
        rows.append(dict(**case,episode=f'{case["engine"]}:{case["family"]}:{case["n"]}:{case["modulation"]}:{seed}',
             seed=seed,waveform=name,frame=int(epoch),method=method,work=r['work'],parts=r['work_parts'],
             iterations=r['iterations'],checks=r['checks'],check_s=r['check_seconds'],
             reference_diff=float(diff.mean()),conservative_diff=conservative,true_ber=float(np.mean(output!=true)),reference_ber=float(np.mean(ref!=true)),
             request_met=r['met_requested_bound'],bound=bound,status=r['status'],coverage=float(r['certified'][:b].mean()),
             reference_unknown=int((~known).sum()),numerical_unknown=int(r['refinement'].get('numerical_unknown',0)),
             violation=bool(r['met_requested_bound'] and conservative>bound+1e-12),fp_certified=r['floating_point_certified'],
             action=t.get('action','none'),valid_envelope=t.get('valid_envelope',False),gamma=t.get('scale',0.),
             epsilon=t.get('delta_c',0.),relative_gamma=t.get('relative_scale',0.),absolute_gamma=t.get('absolute_scale',0.),
             reason=t.get('reason',''),anchor_age=t.get('anchor_age',0),anchor_storage=r['anchor_storage']))
    return rows


def trajectory(stage,case,seed,name,methods,epochs,budget,repetitions=1,keep_trace=False):
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    key=f'{case["engine"]}_{case["family"]}_N{case["n"]}_{case["modulation"]}_{seed}_{name}'
    done=folder/(key+'.complete.json')
    if done.exists():return
    budget.check()
    frames,ys,truth,refs,known,radius=prepare_data(case,seed,name,epochs)
    batches=[];rows=[];bits={};traces={}
    for rep in range(repetitions):
        order=list(methods);rng=np.random.default_rng(seed+rep*91001+['ofdm','otfs','afdm'].index(name));rng.shuffle(order)
        for position,method in enumerate(order):
            budget.check();results,timing=deploy_episode(case,name,frames,ys,methods[method],budget,keep_trace)
            batches.append(dict(**case,seed=seed,waveform=name,method=method,rep=rep,position=position,**timing))
            if rep==0:
                rows+=records(case,seed,name,epochs,method,results,refs,known,truth,methods[method])
                bits[method]=np.array([r['bits'] for r in results])
                if keep_trace:
                    traces[method]=[r['trace'] for r in results]
            else:
                assert np.array_equal(bits[method],np.array([r['bits'] for r in results])),'deterministic repeat outputs changed'
    write(folder/(key+'.json'),rows);write(folder/(key+'.timing.json'),batches)
    np.savez_compressed(folder/(key+'.npz'),received=ys,truth=truth,reference=refs,reference_known=known,reference_radius=radius,
                        **{'bits_'+k:v for k,v in bits.items()})
    if keep_trace:
        # Selected traces are small evaluator evidence, not a deployment input.
        write(folder/(key+'.trace.json'),[{"method":method,"frame":epochs[j],"trace":[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in x.items() if k!='soft'} for x in trace]} for method,alltrace in traces.items() for j,trace in enumerate(alltrace)])
    write(done,dict(source_hash=sources()[0],frames=len(epochs),methods=list(methods),repetitions=repetitions))
    budget.save()
    print(stage,key,'completed',flush=True)


def mechanisms(budget):
    if (OUT/'mechanisms.complete.json').exists():return
    rows=[];rng=np.random.default_rng(77)
    for n in [8,16]:
        ca=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));ca[np.abs(ca)<.8]=0
        ca[-1]=0;a=csr_matrix(ca);nu=.01
        for kind in ['minus','common','row','mismatch','birth','cancelled']:
            for amplitude in ([0.,.001,.01,.1] if kind=='mismatch' else [0.]):
                phase=np.exp(.7j)*np.ones(n) if kind=='common' else np.exp(1j*np.arange(n)*.2) if kind in ['row','mismatch'] else -np.ones(n) if kind=='minus' else np.ones(n)
                ct=phase[:,None]*ca
                if kind=='mismatch':ct+=amplitude*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
                if kind=='birth':ct[-1,0]=.3
                if kind=='cancelled':ct[:]=0;ca0=np.zeros_like(ca);a0=csr_matrix(ca0)
                else:ca0=ca;a0=a
                c=csr_matrix(ct)
                for alignment in ['none','common','row']:
                    d=aligned_change(c,a0,alignment);g=relative_scale(d['epsilon'],nu,nu)
                    aa=ca0.conj().T@ca0+nu*np.eye(n);at=ct.conj().T@ct+nu*np.eye(n)
                    actual=float(eigvalsh(at,aa)[0]);minimum=float(eigvalsh(at-g*aa)[0])
                    rows.append(dict(n=n,kind=kind,amplitude=amplitude,alignment=alignment,epsilon=d['epsilon'],gamma=g,optimal_model_gamma=actual,minimum_loewner=minimum,validator_work=d['work']))
    pd.DataFrame(rows).to_csv(DOC/'phase_mechanisms.csv',index=False)
    natural=[]
    for case in r6.cases():
        if case['n']!=128 or case['modulation']!='qpsk':continue
        w=r6.wave(case,'ofdm');old=r6.frame(case,1080000+100*r6.cases().index(case),0)
        a=old.matrix(w);weights,info=circulant_envelope(a,.01)
        for epoch in [1,10,34,35,55,80,99]:
            current=r6.frame(case,1080000+100*r6.cases().index(case),epoch)
            c=current.matrix(w);path,_=path_change_bound(old,current)
            for alignment in ['none','common','row']:
                d=aligned_change(c,a,alignment);g=relative_scale(d['epsilon'],.01,.01)
                absolute=None if weights is None else float(np.min(1-2*sparse_norm_upper(a)*d['epsilon']/weights))
                natural.append(dict(engine=case['engine'],family=case['family'],epoch=epoch,alignment=alignment,path_epsilon=path,
                                    merged_epsilon=d['epsilon'],relative_gamma=g,absolute_gamma=absolute,anchor_norm_path=channel_norm(old),anchor_norm_merged=sparse_norm_upper(a)))
    pd.DataFrame(natural).to_csv(DOC/'natural_diagnostics.csv',index=False)
    # Old episode inputs are used only as short development mechanism evidence.
    for case in [c for c in r6.cases() if c['n']==128 and c['modulation']=='qpsk' and c['family'] in ['slow','high']]:
        seed=1080000+100*r6.cases().index(case)
        methods={m:specification(m) for m in METHODS+['common_relative','full_r6_deep']}
        for name in ['ofdm','otfs','afdm']:
            trajectory('mechanisms',case,seed,name,methods,[0,1,2,3,4,5],budget,keep_trace=True)
    write(OUT/'mechanisms.complete.json',dict(source_hash=sources()[0],phase_rows=len(rows),natural_rows=len(natural)))


def develop(budget):
    methods={m:specification(m) for m in METHODS+['common_relative','full_r6_deep']}
    for tolerance in [.01,.003,.001,.0003]:methods['residual_'+str(tolerance)]=specification('residual_'+str(tolerance))
    for count in [3,5,7]:methods['stable_'+str(count)]=specification('stable_'+str(count))
    for case in [c for c in r6.cases() if (c['n'],c['modulation']) in [(128,'qpsk'),(256,'16qam')] and c['family'] in ['slow','high']]:
        seed=1170000+100*r6.cases().index(case)
        for name in ['ofdm','otfs','afdm']:trajectory('develop',case,seed,name,methods,list(range(12)),budget,repetitions=2)
    rows=read_stage('develop');choices={}
    for kind in ['residual','stable']:
        candidates=[]
        for method in rows.method.unique():
            if method==kind or method.startswith(kind+'_'):
                sub=rows[rows.method==method].groupby(['engine','family','n','modulation','seed']).agg(diff=('conservative_diff','mean'),work=('work','sum'))
                candidates.append(dict(method=method,mean=float(sub['diff'].mean()),p90=float(sub['diff'].quantile(.9)),work=float(sub.work.mean()),
                                       qualified=bool(sub['diff'].mean()<=.0005 and sub['diff'].quantile(.9)<=.002)))
        qualified=[x for x in candidates if x['qualified']]
        choices[kind]=dict(min(qualified,key=lambda x:x['work']) if qualified else min(candidates,key=lambda x:x['mean']))
        choices[kind]['candidates']=candidates
    write(DOC/'development_selection.json',choices)


def read_stage(stage):
    rows=[]
    for p in sorted((OUT/stage).glob('*.json')):
        if not any(p.name.endswith(x) for x in ['.complete.json','.timing.json','.trace.json']):rows+=json.loads(p.read_text())
    return pd.DataFrame(rows)


def freeze():
    if (CFG/'freeze.json').exists():raise RuntimeError('protocol already frozen')
    choices=json.loads((DOC/'development_selection.json').read_text())
    specs={m:specification(m) for m in METHODS}
    for kind in ['residual','stable']:specs[kind]=specification(choices[kind]['method'])
    code,names=sources();folder=OUT/'sources';folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(folder/(code+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(ROOT/name,name)
    write(CFG/'freeze.json',dict(source_hash=code,hardened_r6='c34f190819c3c1e0ac82157240d41ccef4fc9f52256b7b55c7ff06fa9b9491e0',
         methods=specs,conditions=r6.cases(),repeats=2,frames=100,timing_repetitions=2,seeds=1180000,
         main_delta=.01,secondary_deltas=[0.,.05],validator_policy=asdict(ValidatorPolicy()),
         scale_cases=[dict(engine='tdl',family='slow',n=512,modulation='16qam',cp=96,snr=24.,profile='C',fd=1.,sample_rate=7.68e6,payload_symbols=256),
                      dict(engine='synthetic',family='high',n=512,modulation='qpsk',cp=32,snr=28.,profile='C',fd=2500.,sample_rate=64000.,payload_symbols=512)],
         secondary='N128 QPSK slow/high, engines synthetic/TDL, seed1190000+i*100, three waves, 50 frames, 1 timing repeat; no new independent-main sample',
         primary='Separate same-guarantee efficiency versus legal cached payload-aware Radau/Gray/R6 Deep; heuristic quality/cost and certificate premium separate.',
         reference='full-N white-prior LMMSE; actual N/2 payload for TDL; no reduced-prior reference',
         preconditioner='fresh exact Jacobi, fixed within PCG, all methods',
         quality_equivalence=dict(mean_margin=.00025,p90_margin=.001),no_test_retuning=True,
         timing='whole-episode process/elapsed clock, randomized method order, two independent cache resets, identical-sized return buffers, evaluator/JSON/figures outside',
         maximum_seconds=14400,maximum_bytes=5*1024**3))
    print('Frozen R7',code,choices,flush=True)


def confirm(budget,secondary=False):
    protocol=json.loads((CFG/'freeze.json').read_text())
    if sources()[0]!=protocol['source_hash']:raise RuntimeError('changed frozen source')
    if not secondary:
        for i,case in enumerate(protocol['conditions']):
            for rep in range(protocol['repeats']):
                for name in ['ofdm','otfs','afdm']:
                    trajectory('confirm',case,protocol['seeds']+i*100+rep,name,protocol['methods'],list(range(protocol['frames'])),budget,protocol['timing_repetitions'])
        for i,case in enumerate(protocol['scale_cases']):
            for name in ['ofdm','otfs','afdm']:
                methods={k:v for k,v in protocol['methods'].items() if k in ['gray','radau','r6_deep','phase_relative','relative_radau','residual','stable']}
                trajectory('scale',case,1200000+i*100,name,methods,list(range(50)),budget,2)
    else:
        for i,case in enumerate(protocol['conditions']):
            if case['n']!=128 or case['modulation']!='qpsk' or case['family'] not in ['slow','high']:continue
            for delta in protocol['secondary_deltas']:
                for name in ['ofdm','otfs','afdm']:
                    methods={k:{**v,'delta':delta} for k,v in protocol['methods'].items() if k in ['gray','radau','r6_deep','phase_relative','relative_radau']}
                    trajectory('secondary_'+str(delta),case,1190000+i*100,name,methods,list(range(50)),budget)
    write(OUT/('secondary.complete.json' if secondary else 'confirm.complete.json'),dict(source_hash=sources()[0],completed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mechanisms','develop','freeze','confirm','secondary']);a=p.parse_args();b=Budget()
    try:
        if a.stage=='mechanisms':mechanisms(b)
        elif a.stage=='develop':develop(b)
        elif a.stage=='freeze':freeze()
        else:confirm(b,a.stage=='secondary')
    finally:b.save()
