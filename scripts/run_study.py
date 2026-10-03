"""R5 bounded new evidence; dev -> validation -> freeze -> one confirmation.

All reference computations are evaluator-only. No interpolation substitutes for
running the selected quality-matched baseline on fresh test frames.
"""
import argparse
import os
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator,cg
from waveforge6g.channels.tdl_profile import TDLProfile
from waveforge6g.core.modulation import bits_per_symbol,modulate,demodulate
from waveforge6g.waveforms import create_waveform
from waveforge6g.receivers.certiphy import StopRule,bit_margins
from waveforge6g.receivers.certiphy_deep import DeepRefiner
from waveforge6g.receivers.certiphy_selective import ProfileCertiPHY,SelectiveRefiner,SpectralAblation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/os.environ.get('WAVEFORGE_RUN','certiphy'); DOC=OUT/'analysis'
OUT.mkdir(parents=True,exist_ok=True); DOC.mkdir(parents=True,exist_ok=True)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')


def source_hash():
    paths=sorted((ROOT/'src/waveforge6g').rglob('*.py'))+[Path(__file__),ROOT/'configs/tdl_profiles.json']
    h=hashlib.sha256()
    for p in paths: h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    return h.hexdigest(),paths


def conditions():
    rows=[]
    for p,profile in enumerate('ACE'):
        for speed,fd in enumerate([5.,500.]):
            for level,snr in enumerate([12.,28.]):
                i=len(rows)
                rows.append(dict(condition=f'{profile}_{int(fd)}_{int(snr)}',profile=profile,fd=fd,snr=snr,
                                 n=256 if (p+speed+level)%2==0 else 1024,
                                 modulation='qpsk' if (p+level)%2==0 else '16qam',
                                 ds=100e-9 if (p+speed)%2==0 else 300e-9,cp=128,fs=7.68e6))
    return rows


def spec(method,delta=.01,**kw):
    return dict(method=method,delta=delta,**kw)


def candidates(stage,gate=None):
    methods={}
    for delta in [0.,.01,.05]:
        for method in ['global','radau','gray','deep','selective']:
            methods[f'{method}:{delta:g}']=spec(method,delta,gate=gate or [.01,12.])
    if stage=='develop':
        for limit in [.001,.01,.05]:
            methods[f'gate:{limit}']=spec('selective',.01,gate=[limit,12.])
    for method in ['energy','direction']: methods[f'{method}:0.01']=spec(method)
    for k in [8,16,32,64,128,256]: methods[f'fixed:{k}']=spec('fixed',k=k)
    for tol in [1e-2,3e-3,1e-3,3e-4,1e-4,1e-5]: methods[f'residual:{tol:g}']=spec('residual_fast',tol=tol)
    for stable in [2,4,8,12]: methods[f'stable:{stable}']=spec('stable_fast',stable=stable)
    return methods


def configuration(s):
    method=s['method']; kw=dict(method=method,delta=s['delta'],max_iterations=256,period=4,schedule='adaptive')
    refiner=None
    if method in ['deep','selective','energy','direction']:
        kw['method']='gray'
        if method=='deep': refiner=DeepRefiner('spectral')
        elif method=='selective': refiner=SelectiveRefiner(*s['gate'])
        else: refiner=SpectralAblation(method=='direction')
    if method=='fixed': kw['fixed_iterations']=s['k']
    if method=='residual_fast': kw.update(tolerance=s['tol'],period=2,schedule='periodic')
    if method=='stable_fast': kw.update(stable_checks=s['stable'],period=2,schedule='periodic')
    return StopRule(**kw),refiner


def reference(rx,y):
    c=rx.matrix; noise=rx.noise; b=rx.adjoint@y; n=rx.n
    if n<=256:
        a=(rx.adjoint@c).toarray()+noise*np.eye(n); factor=cho_factor(a)
        z=cho_solve(factor,b)
        cc=c.toarray().astype(np.clongdouble); yy=y.astype(np.clongdouble)
        r=cc.conj().T@(yy-cc@z.astype(np.clongdouble))-noise*z
        z+=cho_solve(factor,np.asarray(r,complex))
        r=cc.conj().T@(yy-cc@z.astype(np.clongdouble))-noise*z
    else:
        op=LinearOperator((n,n),matvec=rx.normal,dtype=complex)
        pre=LinearOperator((n,n),matvec=lambda x:x/rx.diagonal,dtype=complex)
        z,info=cg(op,b,M=pre,rtol=2e-14,atol=0,maxiter=8*n)
        if info: raise RuntimeError('reference convergence failed; retain frame, do not certify')
        r=b-rx.normal(z)
    radius=float(np.linalg.norm(r))/noise+128*np.finfo(float).eps*(1+np.linalg.norm(b)+rx.beta*np.linalg.norm(z))/noise
    soft=rx.wave.analysis(z)
    return soft,bit_margins(soft,rx.modulation).ravel()>radius,radius


class Budget:
    def __init__(self):
        self.path=OUT/'budget.json'; self.before=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else dict(seconds=0.,receivers=0)
        self.start=time.perf_counter(); self.calls=0
    def check(self):
        if self.before['seconds']+time.perf_counter()-self.start>14400: raise RuntimeError('4h experiment cap')
        if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())>5*1024**3: raise RuntimeError('5GiB cap')
    def save(self): write(self.path,dict(seconds=self.before['seconds']+time.perf_counter()-self.start,receivers=self.before['receivers']+self.calls))


def run(stage,budget):
    selection=json.loads((DOC/'development_selection.json').read_text(encoding='utf-8')) if stage!='develop' else None
    gate=None if selection is None else selection['gate']
    methods=candidates(stage,gate)
    seedbase,repeats,frames=(900000,1,1) if stage=='develop' else (910000,2,1)
    if stage=='confirm':
        freeze=json.loads((DOC/'freeze.json').read_text(encoding='utf-8'))
        if source_hash()[0]!=freeze['source_hash']: raise RuntimeError('source differs from frozen code')
        methods=freeze['methods']; seedbase,repeats,frames=920000,4,2
    destination=OUT/stage; destination.mkdir(exist_ok=True)
    if (destination/'complete.json').exists(): raise RuntimeError('completed cohort preserved')
    rows=[]
    for ci,case in enumerate(conditions()):
        for repeat in range(repeats):
            seed=seedbase+ci*100+repeat
            channel=TDLProfile(case['profile'],seed,case['ds'],case['fd'],case['fs'])
            for frame in range(frames):
                rng=np.random.default_rng(seed*10+frame)
                truth=rng.integers(0,2,case['n']*bits_per_symbol(case['modulation']),dtype=np.uint8)
                symbols=modulate(truth,case['modulation']); noise=10**(-case['snr']/10)
                awgn=(rng.normal(size=case['n'])+1j*rng.normal(size=case['n']))*np.sqrt(noise/2)
                start_index=frame*(case['n']+case['cp'])
                for name in ['ofdm','otfs','afdm']:
                    budget.check()
                    key=f'{seed}_{frame}_{name}'; rowpath=destination/(key+'.json')
                    if rowpath.exists(): rows.extend(json.loads(rowpath.read_text(encoding='utf-8'))); continue
                    wave=create_waveform(name,case['n'],case['cp'],16)
                    y=channel.apply(wave.modulate(symbols),start_index)[case['cp']:]+awgn
                    evaluator=ProfileCertiPHY(wave,channel,noise,case['modulation'],start_index)
                    ref,known,ref_radius=reference(evaluator,y); refbits=demodulate(ref,case['modulation'])
                    local=[]; outputs=[]
                    # Predetermined rotation reduces systematic CPU-order confounding.
                    ordered=list(methods); shift=(seed+frame)%len(ordered); ordered=ordered[shift:]+ordered[:shift]
                    for method in ordered:
                        rule,refiner=configuration(methods[method])
                        rx=ProfileCertiPHY(wave,channel,noise,case['modulation'],start_index)
                        cap=rx.setup_work+300*rx.step_work+30e6*rx.n/1024
                        result=rx.solve(y,rule,max_work=cap,refiner=refiner)
                        budget.calls+=1
                        mismatch=result['bits']!=refbits
                        cert_viol=bool(np.sum(mismatch&known)>np.floor(result['disagreement_bound']*len(truth)+1e-7))
                        row=dict(**case,seed=seed,cluster=f'{case["condition"]}:{seed}',frame=frame,waveform=name,method=method,
                                 work=result['work'],cpu=result['process_cpu_s'],wall=result['elapsed_s'],check_s=result['check_seconds'],
                                 memory=result['array_storage_bytes'],iterations=result['iterations'],checks=result['checks'],met=result['met_requested_bound'],
                                 disagreement=float(mismatch.mean()),disagreement_upper=float((mismatch|~known).mean()),ber=float(np.mean(result['bits']!=truth)),
                                 bound=result['disagreement_bound'],coverage=1-result['unknown_fraction'],status=result['status'],violation=cert_viol,
                                 reference_unknown=int((~known).sum()),reference_radius=ref_radius,
                                 numerical_unknown=result['refinement'].get('numerical_unknown',0),
                                 gate_enabled=result['refinement'].get('gate_enabled',False),triggered=result['refinement'].get('triggered',False),
                                 parts=result['work_parts'],refinement=result['refinement'],fp_certified=result['floating_point_certified'])
                        local.append(row); outputs.append(result['bits'])
                    np.savez_compressed(destination/(key+'.npz'),received=y,truth=truth,reference=ref,reference_known=known,
                                        methods=np.array(ordered),outputs=np.array(outputs),channel_seed=seed,start_index=start_index)
                    write(rowpath,local); rows.extend(local)
            print(stage,case['condition'],seed,flush=True)
    df=pd.DataFrame(rows); df.to_csv(DOC/(stage+'.csv'),index=False)
    write(destination/'complete.json',dict(source_hash=source_hash()[0],rows=len(df),methods=methods,seedbase=seedbase,repeats=repeats,frames=frames))
    if stage=='develop': choose_development(df)
    if stage=='validate': freeze_protocol(df,methods)


def quality(df,method):
    subset=df[df.method==method].groupby('cluster').agg(disagreement=('disagreement_upper','mean'),work=('work','mean'))
    return dict(mean=float(subset.disagreement.mean()),tail=float(subset.disagreement.quantile(.9)),work=float(subset.work.mean()))


def choose_development(df):
    options=['selective:0.01']+[f'gate:{v}' for v in [.001,.01,.05]]
    # Development-only gate choice: mean work + 0.25 P95; no test feedback.
    winner=min(options,key=lambda m:df[df.method==m].work.mean()+.25*df[df.method==m].work.quantile(.95))
    gate=[float(winner.split(':')[1]),12.] if winner.startswith('gate') else [.01,12.]
    q1=quality(df,winner); q2=quality(df,'selective:0.05')
    targets=[dict(name='tight',mean=max(.0005,np.ceil(2*q1['mean']/.0005)*.0005),tail=max(.002,np.ceil(2*q1['tail']/.001)*.001)),
             dict(name='loose',mean=max(.002,np.ceil(2*q2['mean']/.0005)*.0005),tail=max(.008,np.ceil(2*q2['tail']/.001)*.001))]
    targets[1]['mean']=max(targets[1]['mean'],4*targets[0]['mean']); targets[1]['tail']=max(targets[1]['tail'],4*targets[0]['tail'])
    write(DOC/'development_selection.json',dict(gate=gate,winner=winner,targets=targets,formula='2x dev empirical mean/P90, upward grid .0005/.001, floors .0005/.002 and .002/.008; loose >=4x tight'))


def freeze_protocol(df,all_methods):
    dev=json.loads((DOC/'development_selection.json').read_text(encoding='utf-8')); selected={}; details=[]
    for target in dev['targets']:
        selected[target['name']]={}
        for family in ['fixed','residual','stable','selective']:
            names=[m for m in all_methods if m.startswith(family+':')]
            qualified=[]
            for m in names:
                q=quality(df,m); ok=q['mean']<=target['mean'] and q['tail']<=target['tail']
                details.append(dict(target=target['name'],family=family,method=m,qualified=ok,**q))
                if ok: qualified.append((q['work'],m))
            selected[target['name']][family]=min(qualified)[1] if qualified else None
    methods={k:v for k,v in all_methods.items() if k.split(':')[0] in ['global','radau','gray','deep','selective','energy','direction']}
    for families in selected.values():
        for m in families.values():
            if m is not None: methods[m]=all_methods[m]
    code,paths=source_hash()
    (OUT/'sources').mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT/'sources'/f'{code}.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(ROOT))
    write(DOC/'freeze.json',dict(source_hash=code,gate=dev['gate'],targets=dev['targets'],selected=selected,methods=methods,
                               conditions=conditions(),confirmation_seedbase=920000,confirmation_repeats=4,confirmation_frames=2,
                               cluster='independent condition/channel seed; all waveforms and frames pooled within cluster',
                               max_iterations=256,work_cap='setup + 300 PCG steps + 30e6*N/1024',
                               primary='selective vs Gray and Radau at delta .01, all 48 clusters; ratio of means and paired cluster bootstrap',
                               no_test_retuning=True,floating_point_certified=False))
    pd.DataFrame(details).to_csv(DOC/'validation_selection.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['develop','validate','confirm']); a=p.parse_args(); budget=Budget()
    try: run(a.stage,budget)
    finally: budget.save()
