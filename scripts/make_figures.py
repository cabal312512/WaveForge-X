"""Read-only R5 aggregation; never calls a receiver or changes frozen thresholds."""
import ast,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_study import ROOT,OUT,DOC,write


def metadata_literal(text):
    class Scalars(ast.NodeTransformer):
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='np' and node.func.attr in ['float64','int64','bool_']
                and len(node.args)==1 and not node.keywords):
                return self.visit(node.args[0])
            raise ValueError('unsupported metadata expression')
    return ast.literal_eval(Scalars().visit(ast.parse(text,mode='eval')))


def clustered(frame,field):
    return frame.groupby(['cluster','method'])[field].mean().unstack()


def paired_ci(df,a,b,field='work',difference=False):
    pair=clustered(df,field)[[a,b]].dropna().to_numpy()
    rng=np.random.default_rng(975001); ix=rng.integers(0,len(pair),(5000,len(pair)))
    means=pair[ix].mean(axis=1)
    samples=means[:,0]-means[:,1] if difference else means[:,0]/means[:,1]
    estimate=float(pair[:,0].mean()-pair[:,1].mean()) if difference else float(pair[:,0].mean()/pair[:,1].mean())
    return dict(estimate=estimate,low=float(np.quantile(samples,.025)),high=float(np.quantile(samples,.975)),clusters=len(pair))


def save(fig,name):
    fig.savefig(DOC/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)


def main():
    df=pd.read_csv(DOC/'confirm.csv'); protocol=json.loads((DOC/'freeze.json').read_text(encoding='utf-8')); summary=[]
    timing=json.loads((DOC/'filter_timing.json').read_text(encoding='utf-8'))
    cpu_extra=[];wall_extra=[]
    for row in df.itertuples():
        t=next(t for t in timing if t['profile']==row.profile and abs(t['ds']-row.ds)<1e-15)
        cpu_extra.append(t['build_cpu_s']/2);wall_extra.append(t['build_wall_s']/2)
    # Distinguish actually measured per-call timing from a common setup amortization.
    df['cpu_with_filter']=df.cpu+cpu_extra;df['wall_with_filter']=df.wall+wall_extra
    for method,g in df.groupby('method'):
        row=dict(method=method,frames=len(g),clusters=g.cluster.nunique(),mean_work=g.work.mean(),mean_cpu_ms=1000*g.cpu.mean(),
                 p95_cpu_ms=1000*g.cpu.quantile(.95),mean_wall_ms=1000*g.wall.mean(),completion=g.met.mean(),
                 disagreement=g.disagreement.mean(),ber=g.ber.mean(),coverage=g.coverage.mean(),iterations=g.iterations.mean(),
                 checks=g.checks.mean(),check_wall_fraction=g.check_s.sum()/g.wall.sum(),violation=int(g.violation.sum()),
                 numerical_unknown=int(g.numerical_unknown.sum()),reference_unknown=int(g.reference_unknown.sum()),
                 gate_rate=g.gate_enabled.mean(),envelope_rate=g.triggered.mean(),memory_resident_max_bytes=g.memory.max())
        row.update(mean_cpu_with_filter_ms=1000*g.cpu_with_filter.mean(),mean_wall_with_filter_ms=1000*g.wall_with_filter.mean())
        row['normal_termination_rate']=float((~g.status.isin(['work_cap','iteration_cap'])).mean())
        if method.startswith(('fixed:','residual:','stable:')):row['completion']=np.nan
        for q in [.5,.9,.95,.99]: row[f'p{int(q*100)}_work']=g.work.quantile(q)
        trajectory=g.groupby('cluster').disagreement.mean();row['p90_trajectory_disagreement']=trajectory.quantile(.9)
        summary.append(row)
    summary=pd.DataFrame(summary).set_index('method'); summary.to_csv(DOC/'summary.csv')
    comparisons={}
    for delta in [0,.01,.05]:
        for baseline in ['gray','radau','deep']:
            a,b=f'selective:{delta:g}',f'{baseline}:{delta:g}'
            comparisons[f'{a}/{b}']=paired_ci(df,a,b)
    write(DOC/'paired_cost.json',comparisons)
    matched=[]
    for target in protocol['targets']:
        selections=protocol['selected'][target['name']]; reference_method=selections['selective']
        for family,method in selections.items():
            row=dict(target=target['name'],family=family,method=method,q=target['mean'],q_tail=target['tail'])
            if method is None:
                row.update(status='no validation-qualified candidate');matched.append(row);continue
            g=df[df.method==method];trajectory=g.groupby('cluster').disagreement_upper.mean()
            row.update(test_mean=trajectory.mean(),test_p90=trajectory.quantile(.9),mean_cost=g.work.mean(),p95_cost=g.work.quantile(.95),completion=g.met.mean())
            row['normal_termination_rate']=float((~g.status.isin(['work_cap','iteration_cap'])).mean())
            if family!='selective':row['completion']=np.nan
            row['target_pass']=bool(row['test_mean']<=target['mean'] and row['test_p90']<=target['tail'])
            if reference_method is not None:
                ref=df[df.method==reference_method].groupby('cluster').disagreement_upper.mean()
                diff=paired_ci(df,method,reference_method,'disagreement_upper',True)
                equivalent=(diff['low']>=-target['mean']/2 and diff['high']<=target['mean']/2 and abs(row['test_p90']-ref.quantile(.9))<=target['tail']/2)
                transfer=row['target_pass'] and ref.mean()<=target['mean'] and ref.quantile(.9)<=target['tail'] and equivalent
                row.update(diff_ci_low=diff['low'],diff_ci_high=diff['high'],equivalent=bool(equivalent),status='matched within frozen tolerance' if transfer else 'quality matching did not transfer')
                if transfer: row.update({f'baseline_over_selective_{k}':v for k,v in paired_ci(df,method,reference_method).items()})
            matched.append(row)
    pd.DataFrame(matched).to_csv(DOC/'quality_test.csv',index=False)
    ablation=['gray:0.01','energy:0.01','direction:0.01','deep:0.01','selective:0.01']
    component=[]
    for method in ablation:
        g=df[df.method==method]; parts=[ast.literal_eval(x) for x in g.parts]
        prep=np.mean([sum(v for k,v in p.items() if k in ['preparation','deep_envelope_preparation','deep_inverse_directions','selective_gate']) for p in parts])
        iteration=np.mean([p.get('pcg',0)+p.get('iteration',0) for p in parts])
        # Ledger uses 'pcg_iterations'; retain actual key for transparent grouping.
        if iteration==0: iteration=np.mean([sum(v for k,v in p.items() if 'iteration' in k or k=='pcg_step') for p in parts])
        component.append(dict(method=method,preparation=prep,iteration=iteration,remaining=g.work.mean()-prep-iteration,
                              mean_iterations=g.iterations.mean(),mean_checks=g.checks.mean(),completion=g.met.mean(),
                              p95=g.work.quantile(.95),refinement_successes=sum(metadata_literal(x).get('successes',0) for x in g.refinement)))
    pd.DataFrame(component).to_csv(DOC/'ablation.csv',index=False)
    cond=[]
    for name,g in df.groupby('condition'):
        a=g[g.method=='selective:0.01'];b=g[g.method=='gray:0.01']
        pair=clustered(g,'work')[['selective:0.01','gray:0.01']]
        ratios=pair.iloc[:,0]/pair.iloc[:,1]
        cond.append(dict(condition=name,ratio=a.work.mean()/b.work.mean(),minimum_cluster_ratio=ratios.min(),maximum_cluster_ratio=ratios.max(),
                         completion=a.met.mean(),gate_rate=a.gate_enabled.mean(),mean_cpu_ms=1000*a.cpu.mean()))
    pd.DataFrame(cond).to_csv(DOC/'conditions.csv',index=False)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    colors=['#7b8794','#d49b33','#35699b','#a34b64','#238476']
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for color,family in zip(colors,['global','radau','gray','deep','selective']):
        rows=summary.loc[[f'{family}:{d:g}' for d in [0,.01,.05]]]
        for ax,field,scale,label in zip(axes.ravel(),['mean_work','p95_work','completion','mean_cpu_with_filter_ms'],[1e-6,1e-6,100,1],['Mean work (M)','P95 work (M)','Requests completed (%)','CPU incl. FIR setup (ms)']):
            ax.plot([0,1,2],rows[field]*scale,'o-',color=color,label=family);ax.set(xticks=[0,1,2],xticklabels=['0','.01','.05'],xlabel='Disagreement budget',ylabel=label)
    axes[1,0].set(ylim=(0,105),yticks=[0,50,100])
    axes[0,0].legend(ncol=2,fontsize=8);fig.tight_layout();save(fig,'guarantee_cost')
    fig,ax=plt.subplots(figsize=(8,5));seen=set()
    palette={'gray':'#35699b','deep':'#a34b64','selective':'#238476','fixed':'#7b8794','residual':'#d49b33','stable':'#7059a4'}
    for method,row in summary.iterrows():
        family=method.split(':')[0]
        if family in ['energy','direction','global','radau']:continue
        marker='o' if family in ['gray','deep','selective'] else 'x'
        ax.scatter(1e4*row.disagreement,row.mean_work/1e6,marker=marker,s=45,color=palette[family],label=family if family not in seen else None)
        seen.add(family)
        label=method.split(':')[1]
        dy=-13 if family in ['selective','stable'] else 8
        ax.annotate(label,(1e4*row.disagreement,row.mean_work/1e6),fontsize=8,xytext=(5,dy),textcoords='offset points',color=palette[family])
    ax.legend(title='Labels: delta / K / tolerance / stable checks',fontsize=8,loc='upper right')
    ax.set(xlabel='Observed reference disagreement (x 1e-4)',ylabel='Mean total work (M)');fig.tight_layout();save(fig,'quality_cost')
    comp=pd.DataFrame(component);fig,axes=plt.subplots(1,2,figsize=(10,4));bottom=np.zeros(5)
    for col,color in zip(['preparation','iteration','remaining'],['#7b8794','#35699b','#d49b33']):
        axes[0].bar(np.arange(5),comp[col]/1e6,bottom=bottom,color=color,label=col);bottom+=comp[col].to_numpy()/1e6
    labels=['Gray','+energy','+direction','Deep','selective']
    axes[0].set(xticks=range(5),xticklabels=labels,ylabel='Mean work (M)');axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(5),comp.mean_iterations,color=colors);axes[1].set(xticks=range(5),xticklabels=labels,ylabel='Mean PCG iterations')
    fig.tight_layout();save(fig,'ablation_cost')
    cond=pd.DataFrame(cond);fig,ax=plt.subplots(figsize=(9,4));pos=np.arange(len(cond))
    ax.bar(pos,cond.ratio-1,bottom=1,color=['#238476' if x<1 else '#a34b64' for x in cond.ratio])
    for i,r in cond.iterrows():ax.plot([i,i],[r.minimum_cluster_ratio,r.maximum_cluster_ratio],color='.25',lw=1)
    ax.axhline(1,color='.4',ls='--');ax.set(xticks=pos,xticklabels=cond.condition,ylabel='Selective / Gray total work',title='Four channel clusters per condition; lines show observed range')
    ax.tick_params(axis='x',rotation=50);fig.tight_layout();save(fig,'condition_cost')
    diag=json.loads((DOC/'diagnostics.json').read_text(encoding='utf-8'));fig,axes=plt.subplots(1,2,figsize=(10,4));response=pd.DataFrame(diag['fractional_response'])
    axes[0].plot(response.delay[:5],response.rms_full[:5],'o-',label='Full Nyquist band');axes[0].plot(response.delay[:5],response.rms_80percent[:5],'s-',label='Central 80%')
    axes[0].set(xlabel='Fractional delay (samples)',ylabel='RMS complex response error');axes[0].legend()
    d=diag['doppler'];axes[1].plot(d['lags'],d['jakes'],'-',label='Jakes');axes[1].plot(d['lags'],d['measured_real'],'o',label='SOS ensemble')
    axes[1].set(xlabel='Lag (samples)',ylabel='Normalized real correlation');axes[1].legend();fig.tight_layout();save(fig,'channel_approximation')
    # Save reproducibility hashes of all raw evidence; no packaging or release.
    import hashlib
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='raw_manifest.json']
    write(OUT/'raw_manifest.json',{str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    print(summary.loc[['gray:0.01','radau:0.01','deep:0.01','selective:0.01']].to_string())
    print(pd.DataFrame(matched).to_string(index=False))


if __name__=='__main__':main()
