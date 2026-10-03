"""Measure common immutable FIR table construction, amortized over two frames.

This microbenchmark does not choose/tune receiver parameters. Primary receiver
timings already include each full SOS/COO/CSR rebuild, never shared between methods.
"""
import time
import numpy as np
from run_study import OUT,DOC,Budget,write,conditions
from waveforge6g.channels.tdl_profile import TDLProfile,fractional_kernel


if __name__=='__main__':
    if (OUT/'filter_timing.json').exists(): raise RuntimeError('timing evidence preserved')
    budget=Budget();rows=[]
    try:
        pairs=sorted({(c['profile'],c['ds']) for c in conditions()})
        for profile,ds in pairs:
            c=TDLProfile(profile,977000,delay_spread_s=ds)
            samples=[]
            for block in range(8):
                start,cpu=time.perf_counter(),time.process_time()
                for repeat in range(50):
                    kernels=[fractional_kernel(d,16) for d in c.delays]
                samples.append(dict(wall=(time.perf_counter()-start)/50,cpu=(time.process_time()-cpu)/50))
            rows.append(dict(profile=profile,ds=ds,build_cpu_s=float(np.mean([s['cpu'] for s in samples])),
                             build_wall_s=float(np.median([s['wall'] for s in samples])),samples=samples,
                             amortized_frames=2,coefficient_storage_bytes=sum(d.nbytes+h.nbytes for d,h in kernels)))
        write(OUT/'filter_timing.json',rows);write(DOC/'filter_timing.json',rows)
    finally:budget.save()
