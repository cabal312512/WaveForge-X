"""Small post-hoc timing diagnostic; never substitutes frozen primary results."""
import json
import pandas as pd
import research_v6 as r
from waveforge6g.receivers.certiphy_reuse import ReusePolicy


def main():
    marker=r.OUT/'timing_audit/complete.json'
    if marker.exists():raise RuntimeError('Completed timing diagnostic preserved')
    p=json.loads((r.CFG/'freeze.json').read_text());budget=r.Budget();rows=[]
    methods={k:v for k,v in p['methods'].items() if k.startswith('reuse_')}
    selected=[c for c in p['conditions'] if c['n']==256 and c['modulation']=='16qam' and c['family'] in ['slow','high']]
    try:
        for i,c in enumerate(selected):
            seed=1100000+i*100
            order=['fresh','lagged'] if i%2==0 else ['lagged','fresh']
            for policy in order:
                age=1 if policy=='fresh' else 20
                st=ReusePolicy(**{**p['policy'],'diagonal_age':age})
                raw=r.run_trajectory('timing_'+policy,c,seed,'ofdm',methods,st,30,budget)
                rows += [dict(**x,timing_policy=policy) for x in raw]
            budget.save()
        df=pd.DataFrame(rows);df.to_csv(r.DOC/'timing_audit.csv',index=False)
        group=df.groupby(['engine','family','episode','method','timing_policy']).agg(work=('work','sum'),cpu=('cpu','sum'),wall=('wall','sum'),iterations=('iterations','mean'),jacobi_rebuild=('jacobi_rebuilt','mean'),disagreement=('disagreement','mean')).reset_index()
        result=[]
        for keys,s in group.groupby(['engine','family','episode','method']):
            q=s.set_index('timing_policy');a,b=q.loc['fresh'],q.loc['lagged']
            result.append(dict(zip(['engine','family','episode','method'],keys),
                work_ratio=b.work/a.work,cpu_ratio=b.cpu/a.cpu if a.cpu>0 else None,wall_ratio=b.wall/a.wall,
                fresh_cpu=a.cpu,lagged_cpu=b.cpu,fresh_wall=a.wall,lagged_wall=b.wall,
                fresh_rebuild=a.jacobi_rebuild,lagged_rebuild=b.jacobi_rebuild,
                iteration_change=b.iterations-a.iterations,difference_change=b.disagreement-a.disagreement))
        pd.DataFrame(result).to_csv(r.DOC/'timing_audit_summary.csv',index=False)
        r.write(marker,dict(receiver_calls=len(df),clusters=df.episode.nunique(),frames=30,waveform='ofdm',source_hash=r.sources()[0],
            purpose='post-hoc diagnostic for anomalous ablation CPU ratio; fresh/lagged run in one process with small equally sized episode buffers, batch order alternates, common received data; no tuning or new primary hypothesis test'))
    finally:budget.save()


if __name__=='__main__':main()
