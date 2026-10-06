"""Bounded reuse experiments: diagnostics -> develop -> freeze -> confirm.

Current received signals, complete models and independent episode pairing are
shared. Each method owns its deployment cache. References are evaluator-only.
"""
import argparse,hashlib,json,time,zipfile,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from waveforge6g.channels.reuse_paths import synthetic_frame,tdl_frame,accurate_kernel,OversampledWave
from waveforge6g.receivers.certiphy_reuse import ReuseCertiPHY,ReuseRefiner,ReuseState,ReusePolicy,ColdPathCertiPHY,ColdRefiner
from waveforge6g.receivers.certiphy import StopRule,bit_margins,WorkLedger
from waveforge6g.core.modulation import bits_per_symbol,modulate,demodulate
from waveforge6g.waveforms import create_waveform

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/research_v6';DOC=ROOT/'docs/research_v6';CFG=ROOT/'configs/research_v6'
for p in [OUT,DOC,CFG]:p.mkdir(parents=True,exist_ok=True)


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8');tmp.replace(p)


def sources():
    names=['src/waveforge6g/channels/reuse_paths.py','src/waveforge6g/channels/tdl_profile.py',
           'src/waveforge6g/receivers/certiphy_reuse.py','src/waveforge6g/receivers/reuse_geometry.py',
           'src/waveforge6g/receivers/certiphy.py','src/waveforge6g/receivers/certiphy_deep.py','src/waveforge6g/receivers/certiphy_selective.py',
           'src/waveforge6g/receivers/budgeted.py','scripts/research_v6.py','configs/research_v5/tdl_profiles.json']
    names+=['src/waveforge6g/waveforms/'+p.name for p in sorted((ROOT/'src/waveforge6g/waveforms').glob('*.py'))]
    h=hashlib.sha256()
    for n in names:h.update(n.encode());h.update((ROOT/n).read_bytes())
    return h.hexdigest(),names


class Budget:
    def __init__(self):
        self.old=json.loads((OUT/'budget.json').read_text()) if (OUT/'budget.json').exists() else dict(seconds=0.,receiver_calls=0,replay_preparations=0)
        self.start=time.perf_counter();self.calls=self.replays=0
    def check(self):
        if self.old['seconds']+time.perf_counter()-self.start>=14400:raise RuntimeError('four-hour new-experiment cap; completed trajectories retained')
    def save(self):
        sizes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        write(OUT/'budget.json',dict(seconds=self.old['seconds']+time.perf_counter()-self.start,receiver_calls=self.old['receiver_calls']+self.calls,replay_preparations=self.old.get('replay_preparations',0)+self.replays,bytes=sizes))
        if sizes>5*1024**3:raise RuntimeError('5GiB cap')


def cases():
    rows=[]
    for engine in ['synthetic','tdl']:
        for family in ['slow','moderate','high','abrupt']:
            for n in [128,256]:
                for mod in ['qpsk','16qam']:
                    profile={ (128,'qpsk'):'C',(128,'16qam'):'E',(256,'qpsk'):'A',(256,'16qam'):'C'}[(n,mod)]
                    rows.append(dict(engine=engine,family=family,n=n,modulation=mod,cp=96 if engine=='tdl' else 32,
                                     snr=28. if mod=='qpsk' else 24.,profile=profile,fd={'slow':1.,'moderate':100.,'high':2500.,'abrupt':5.}[family],
                                     sample_rate=7.68e6 if engine=='tdl' else 64000.,payload_symbols=n//2 if engine=='tdl' else n))
    return rows


def frame(case,seed,epoch):
    if case['engine']=='synthetic':return synthetic_frame(seed,case['n'],case['cp'],epoch,case['family'])
    return tdl_frame(case['profile'],seed,case['n'],case['cp'],epoch,case['fd'],birth=case['family']=='abrupt')


def wave(case,name):
    n,cp=case['n'],case['cp']
    if case['engine']=='tdl':return OversampledWave(create_waveform(name,n//2,cp//2,16),n,cp)
    return create_waveform(name,n,cp,16)


def reference(rx,y):
    # Small dense solve is ONLY in this evaluator; no such object enters state.
    a=(rx.adjoint@rx.matrix).toarray()+rx.noise*np.eye(rx.n);factor=cho_factor(a)
    b=rx.adjoint@y;z=cho_solve(factor,b)
    cc=rx.matrix.toarray().astype(np.clongdouble);yy=y.astype(np.clongdouble)
    r=cc.conj().T@(yy-cc@z.astype(np.clongdouble))-rx.noise*z
    z+=cho_solve(factor,np.asarray(r,complex));r=cc.conj().T@(yy-cc@z.astype(np.clongdouble))-rx.noise*z
    radius=float(np.linalg.norm(r))/rx.alpha+128*np.finfo(float).eps*(1+np.linalg.norm(b)+rx.beta*np.linalg.norm(z))/rx.alpha
    s=rx.wave.analysis(z)
    return s,bit_margins(s,rx.modulation).ravel()>radius,radius


def method_specs(tolerance=.001,delta=.01):
    specs={name:dict(kind=kind,cold=cold,delta=delta,tolerance=tolerance) for name,kind,cold in [
        ('cold_gray','gray',True),('cold_deep','deep',True),('reuse_gray','gray',False),
        ('reuse_deep','deep',False),('reuse_radau','radau',False),('reuse_residual','residual_fast',False)]}
    for k in ['gray','deep']:specs['reset_'+k]=dict(kind=k,cold=False,delta=delta,tolerance=tolerance,reset=True)
    return specs


def rule(spec,case):
    delta=spec['delta']*case['payload_symbols']/case['n']
    return StopRule(method='gray' if spec['kind']=='deep' else spec['kind'],delta=delta,
                    max_iterations=256,period=2 if spec['kind']=='residual_fast' else 4,
                    schedule='periodic' if spec['kind']=='residual_fast' else 'adaptive',tolerance=spec['tolerance'])


def diagnostic(budget):
    dest=OUT/'diagnostics.json'
    if dest.exists():raise RuntimeError('completed diagnostics preserved')
    # Accuracy is evaluated on ALL of the declared occupied band, both endpoints.
    omega=np.linspace(-np.pi/2,np.pi/2,8193);rows=[]
    for frac in np.linspace(0,.99,25):
        budget.check();d,h=accurate_kernel(frac,32);e=np.exp(-1j*omega[:,None]*d)
        response=e@h;ideal=np.exp(-1j*omega*(32+frac));derivative=e@(-1j*d*h)
        # Independent longer sinc model with a stronger window, latency aligned.
        j=np.arange(257);hh=np.sinc(j-128-frac)*np.kaiser(257,16.);hh/=hh.sum()
        long=np.exp(-1j*omega[:,None]*j)@hh*np.exp(1j*omega*96)
        rows.append(dict(fraction=float(frac),rms=float(np.sqrt(np.mean(abs(response-ideal)**2))),maximum=float(abs(response-ideal).max()),
                         group_delay_max_samples=float(abs(-np.imag(derivative/response)-(32+frac)).max()),
                         long_reference_rms=float(np.sqrt(np.mean(abs(long-ideal)**2))),long_reference_max=float(abs(long-ideal).max()),
                         main_vs_long_rms=float(np.sqrt(np.mean(abs(response-long)**2)))))
    if max(r['maximum'] for r in rows)>1e-4 or max(r['group_delay_max_samples'] for r in rows)>.002:raise RuntimeError('main occupied-band FIR accuracy gate failed')
    legacy=pd.read_csv(DOC.parent/'research_v5/ablation.csv') if (DOC.parent/'research_v5/ablation.csv').exists() else pd.read_csv(ROOT/'build/github/data/tables/ablation.csv')
    write(dest,dict(fractional_delay=rows,main_kernel=dict(taps=65,half=32,beta=10.,normalization='DC gain',occupied_band_radians=[-np.pi/2,np.pi/2],oversampling=2,
               payload_symbols='N/2; remaining N/2 zero coordinates, all methods',guard=96,reference='257-tap beta16 DC-normalized, latency aligned; ideal continuous frequency response'),
               legacy_cost=legacy.to_dict(orient='records'),legacy_preserved=True))
    print('accuracy gate passed',max(r['maximum'] for r in rows),max(r['group_delay_max_samples'] for r in rows),flush=True)


def run_trajectory(stage,case,seed,name,methods,policy,frames,budget):
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    key=f'{case["engine"]}_{case["family"]}_N{case["n"]}_{case["modulation"]}_{seed}_{name}'
    rp=folder/(key+'.json');npz=folder/(key+'.npz');done=folder/(key+'.complete.json')
    if done.exists():return json.loads(rp.read_text(encoding='utf-8'))
    old=json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else []
    completed=0 if not old else max(r['frame'] for r in old)+1
    states={m:ReuseState(policy) for m,s in methods.items() if not s['cold']}
    w=wave(case,name);noise=10**(-case['snr']/10);m=case['payload_symbols'];b=m*bits_per_symbol(case['modulation'])
    ys=[];truths=[];refs=[];knowns=[];outputs=[]
    if completed:
        raw=np.load(npz,allow_pickle=False)
        ys=list(raw['received']);truths=list(raw['truth']);refs=list(raw['reference']);knowns=list(raw['reference_known']);outputs=list(raw['outputs'])
        # Rebuild only past CSI cache transitions; no saved payload is solved.
        for epoch in range(completed):
            c=frame(case,seed,epoch)
            for method,st in states.items():
                rx=ReuseCertiPHY(w,c,noise,case['modulation'],st)
                if methods[method]['kind']=='deep':ReuseRefiner(st).prepare(rx,WorkLedger(np.inf))
                budget.replays+=1
    rows=list(old)
    try:
        for epoch in range(completed,frames):
            budget.check();c=frame(case,seed,epoch);rng=np.random.default_rng(seed*1000+epoch)
            truth=rng.integers(0,2,b,dtype=np.uint8)
            s=np.r_[modulate(truth,case['modulation']),np.zeros(case['n']-m,complex)]
            y=c.apply(w.modulate(s))[case['cp']:]+(rng.normal(size=case['n'])+1j*rng.normal(size=case['n']))*np.sqrt(noise/2)
            evaluator=ColdPathCertiPHY(w,c,noise,case['modulation']);ref,known,ref_radius=reference(evaluator,y)
            refbits=demodulate(ref[:m],case['modulation']);known=known[:b]
            ordered=list(methods);shift=(seed+epoch)%len(ordered);ordered=ordered[shift:]+ordered[:shift]
            out={}
            for method in ordered:
                spec=methods[method]
                if spec.get('reset'):states[method]=ReuseState(policy)
                rx=ColdPathCertiPHY(w,c,noise,case['modulation']) if spec['cold'] else ReuseCertiPHY(w,c,noise,case['modulation'],states[method])
                refiner=(ColdRefiner('spectral') if spec['cold'] else ReuseRefiner(states[method])) if spec['kind']=='deep' else None
                result=rx.solve(y,rule(spec,case),max_work=rx.setup_work+512*rx.step_work+30e6,refiner=refiner)
                budget.calls+=1;bits=result['bits'][:b];out[method]=bits
                diff=bits!=refbits;payload_bound=min(1.,result['disagreement_bound']*case['n']/m)
                reuse=result.get('reuse',{});refine=result['refinement']
                audit_eigenvalue=None
                if (refiner is not None and refiner.weights is not None and case['n']==128 and case['modulation']=='qpsk'
                        and seed%100==0 and epoch in [0,1,34,35,54,55,79,80,99]):
                    # Predeclared evaluator-only small-system Loewner attack.
                    ff=np.fft.fft(np.eye(rx.n),axis=0,norm='ortho')
                    aa=(rx.adjoint@rx.matrix).toarray()+noise*np.eye(rx.n)
                    lower=ff.conj().T@(refiner.weights[:,None]*ff)
                    audit_eigenvalue=float(np.linalg.eigvalsh(aa-lower).min())
                    if audit_eigenvalue < -1e-8:raise RuntimeError('unsafe reused/new envelope; blocking bug')
                # A violation is a blocking bug, retained as a failing raw record.
                violation=int(np.sum(diff&known))>int(np.floor(payload_bound*b+1e-7))
                rows.append(dict(**case,episode=f'{case["engine"]}:{case["family"]}:{case["n"]}:{case["modulation"]}:{seed}',seed=seed,waveform=name,frame=epoch,method=method,
                    work=result['work'],cpu=result['process_cpu_s'],wall=result['elapsed_s'],check_seconds=result['check_seconds'],memory=result['array_storage_bytes'],
                    iterations=result['iterations'],checks=result['checks'],met=result['met_requested_bound'],status=result['status'],
                    bound=payload_bound,disagreement=float(diff.mean()),disagreement_upper=float((diff|~known).mean()),ber=float(np.mean(bits!=truth)),reference_ber=float(np.mean(refbits!=truth)),
                    coverage=float(result['certified'][:b].mean()),reference_unknown=int((~known).sum()),reference_radius=ref_radius,violation=violation,
                    numerical_unknown=refine.get('numerical_unknown',0),fp_certified=result['floating_point_certified'],
                    envelope_audit_min_eigenvalue=audit_eigenvalue,
                    action=reuse.get('action','cold' if spec['cold'] else 'static_only'),valid_envelope=refine.get('triggered',False),
                    reason=reuse.get('reason',''),scale=reuse.get('scale',0.),delta_c=reuse.get('delta_c',0.),delta_a_bound=reuse.get('delta_a_norm_bound',0.),
                    static_rebuilt=reuse.get('static_rebuilt',spec['cold']),jacobi_rebuilt=reuse.get('jacobi_rebuilt',True),parts=result['work_parts'],cache=result.get('cache_metrics',{})))
                if violation:raise RuntimeError('observed certificate violation; inspect retained records')
            ys.append(y);truths.append(truth);refs.append(ref);knowns.append(known);outputs.append(np.array([out[k] for k in methods]))
            if (epoch+1)%10==0:
                np.savez_compressed(npz,received=np.array(ys),truth=np.array(truths),reference=np.array(refs),reference_known=np.array(knowns),outputs=np.array(outputs),methods=np.array(list(methods)))
                write(rp,rows)
    finally:
        if len(outputs):
            np.savez_compressed(npz,received=np.array(ys),truth=np.array(truths),reference=np.array(refs),reference_known=np.array(knowns),outputs=np.array(outputs),methods=np.array(list(methods)))
            write(rp,rows)
    write(done,dict(frames=frames,source_hash=sources()[0],methods=methods,policy=asdict(policy)))
    print(stage,key,'frames',frames,flush=True)
    return rows


def development(budget):
    if (OUT/'develop_corrected/complete.json').exists():raise RuntimeError('completed development preserved')
    selected=[c for c in cases() if c['family'] in ['slow','high'] and c['n']==128]
    methods=method_specs()
    for tol in [.01,.003,.0003,.0001]:methods['residual_'+str(tol)]=dict(kind='residual_fast',cold=False,delta=.01,tolerance=tol)
    # Only valid positive scales differ; selected on development episodes, not test.
    for scale in [.25,.75]:methods['deep_scale_'+str(scale)]=dict(kind='deep',cold=False,delta=.01,tolerance=.001,minimum_scale=scale)
    rows=[]
    for i,c in enumerate(selected):
        for name in ['ofdm','otfs','afdm']:
            # For explicit scale variants, one trajectory invocation per policy
            # avoids assigning one mutable cache to counterfactual configurations.
            base={k:v for k,v in methods.items() if not k.startswith('deep_scale')}
            rows+=run_trajectory('develop_corrected',c,1065000+i*100,name,base,ReusePolicy(diagonal_age=1),12,budget)
            for scale in [.25,.75]:
                variant={'deep_scale_'+str(scale):dict(kind='deep',cold=False,delta=.01,tolerance=.001)}
                rows+=run_trajectory('develop_corrected_scale_'+str(scale),c,1065000+i*100,name,variant,ReusePolicy(minimum_scale=scale,diagonal_age=1),12,budget)
    df=pd.DataFrame(rows);df.to_csv(DOC/'develop_corrected.csv',index=False)
    choices=[]
    for method in ['reuse_deep','deep_scale_0.25','deep_scale_0.75']:
        s=df[df.method==method];choices.append(dict(method=method,scale=.5 if method=='reuse_deep' else float(method.split('_')[-1]),score=float(s.work.mean()+.25*s.work.quantile(.95))))
    policy=ReusePolicy(minimum_scale=min(choices,key=lambda x:x['score'])['scale'],diagonal_age=1)
    write(DOC/'development_selection_corrected.json',dict(policy=asdict(policy),scores=choices,quality_targets=dict(mean=.0005,p90=.002),source_hash=sources()[0]))
    write(OUT/'develop_corrected/complete.json',dict(rows=len(rows),new_received_frames=8*3*12,policy=asdict(policy)))


def freeze(budget):
    if (CFG/'freeze.json').exists():raise RuntimeError('frozen protocol preserved')
    dev=json.loads((DOC/'development_selection_corrected.json').read_text());policy=ReusePolicy(**dev['policy'])
    selected=[c for c in cases() if c['family'] in ['slow','moderate','high','abrupt'] and ((c['n']==128 and c['modulation']=='qpsk') or (c['n']==256 and c['modulation']=='16qam'))]
    methods=method_specs()
    for tol in [.01,.003,.0003,.0001]:methods['residual_'+str(tol)]=dict(kind='residual_fast',cold=False,delta=.01,tolerance=tol)
    rows=[]
    for i,c in enumerate(selected):
        for name in ['ofdm','otfs','afdm']:rows+=run_trajectory('validate',c,1070000+i*100,name,methods,policy,10,budget)
    df=pd.DataFrame(rows);df.to_csv(DOC/'validate.csv',index=False);detail=[]
    for method in ['reuse_residual','residual_0.01','residual_0.003','residual_0.0003','residual_0.0001']:
        s=df[df.method==method].groupby('episode').agg(diff=('disagreement_upper','mean'),work=('work','sum'))
        detail.append(dict(method=method,tolerance=methods[method]['tolerance'],mean=float(s['diff'].mean()),p90=float(s['diff'].quantile(.9)),work=float(s.work.mean()),qualified=bool(s['diff'].mean()<=.0005 and s['diff'].quantile(.9)<=.002)))
    qualified=[x for x in detail if x['qualified']]
    winner=min(qualified,key=lambda x:x['work']) if qualified else min(detail,key=lambda x:x['mean'])
    code,names=sources();snapshot=OUT/'sources';snapshot.mkdir(exist_ok=True)
    with zipfile.ZipFile(snapshot/(code+'.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(ROOT/n,n)
    protocol=dict(source_hash=code,policy=asdict(policy),methods=method_specs(winner['tolerance']),residual_selection=winner,validation_candidates=detail,
                  conditions=cases(),confirmation_seedbase=1080000,repeats=3,frames=100,checkpoints=[10,50,100],primary_delta=.01,secondary_deltas=[0.,.05],
                  quality_targets=dict(mean=.0005,p90=.002,equivalence_mean_margin=.00025,equivalence_p90_margin=.001),
                  primary='paired independent episode total work, warm Deep vs cold Gray and warm Radau/residual, primary delta .01',
                  cluster='engine/family/N/modulation/channel seed; all waveforms and frames correlated',bootstrap_repetitions=5000,
                  physical_model='TDL 65-tap DC-normalized, 2x common oversampling with N/2 payload, cp96; complete physical operator',
                  preconditioner='all primary methods fresh exact Jacobi every frame; separate ablation uses shared age20 lagged SPD policy',
                  no_payload_warm_start=True,no_test_retuning=True,floating_point_certified=False,
                  initial_development_excluded='Oversampled OFDM weighted Gray costs initially used native rather than symbol-order weights; corrected before fresh development and protocol freeze. Initial evidence retained, not used for selection.',
                  maximum_seconds=14400,maximum_bytes=5*1024**3)
    write(CFG/'freeze.json',protocol);write(OUT/'validate/complete.json',dict(rows=len(rows),source_hash=code))
    print('Frozen policy',protocol['policy'],'residual',winner,flush=True)


def confirmation(budget,secondary=False):
    protocol=json.loads((CFG/'freeze.json').read_text());code=sources()[0]
    if code!=protocol['source_hash']:raise RuntimeError('frozen implementation differs; do not retune confirmation')
    stage='secondary' if secondary else 'confirm'
    if (OUT/stage/'complete.json').exists():raise RuntimeError('completed confirmation preserved')
    rows=[];policy=ReusePolicy(**protocol['policy'])
    conditions=protocol['conditions']
    if secondary:conditions=[c for c in conditions if c['n']==256 and c['modulation']=='16qam']
    for i,c in enumerate(conditions):
        ci=protocol['conditions'].index(c)
        for rep in range(1 if secondary else protocol['repeats']):
            seed=protocol['confirmation_seedbase']+ci*100+rep
            for name in ['ofdm','otfs','afdm']:
                if secondary:
                    for delta in protocol['secondary_deltas']:
                        methods={k:v for k,v in method_specs(protocol['residual_selection']['tolerance'],delta).items() if k not in ['reuse_residual','reset_gray','reset_deep']}
                        rows+=run_trajectory(stage+'_'+str(delta),c,seed,name,methods,policy,100,budget)
                else:rows+=run_trajectory(stage,c,seed,name,protocol['methods'],policy,100,budget)
            budget.save()
    pd.DataFrame(rows).to_csv(DOC/(stage+'.csv'),index=False)
    write(OUT/stage/'complete.json',dict(rows=len(rows),source_hash=code,paired_episodes=int(pd.DataFrame(rows).episode.nunique()),secondary=secondary))


def precondition_ablation(budget):
    protocol=json.loads((CFG/'freeze.json').read_text())
    if sources()[0]!=protocol['source_hash']:raise RuntimeError('frozen source differs')
    if (OUT/'precondition/complete.json').exists():raise RuntimeError('completed ablation preserved')
    policy=ReusePolicy(**{**protocol['policy'],'diagonal_age':20})
    methods={k:v for k,v in protocol['methods'].items() if k.startswith('reuse_')};rows=[]
    for ci,c in enumerate(protocol['conditions']):
        if c['n']!=256 or c['modulation']!='16qam':continue
        seed=protocol['confirmation_seedbase']+ci*100
        for name in ['ofdm','otfs','afdm']:rows+=run_trajectory('precondition',c,seed,name,methods,policy,100,budget)
    pd.DataFrame(rows).to_csv(DOC/'precondition.csv',index=False)
    write(OUT/'precondition/complete.json',dict(rows=len(rows),source_hash=sources()[0],policy=asdict(policy)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['diagnostics','develop','freeze','confirm','secondary','precondition']);arg=a.parse_args();budget=Budget()
    try:
        if arg.stage=='diagnostics':diagnostic(budget)
        elif arg.stage=='develop':development(budget)
        elif arg.stage=='freeze':freeze(budget)
        elif arg.stage=='precondition':precondition_ablation(budget)
        else:confirmation(budget,arg.stage=='secondary')
    finally:budget.save()
