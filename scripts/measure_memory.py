"""Separate traced-allocation diagnostic; excluded from primary CPU timing."""
import tracemalloc
import numpy as np
from run_study import OUT,DOC,Budget,write,configuration
from waveforge6g.channels.tdl_profile import TDLProfile
from waveforge6g.receivers.certiphy_selective import ProfileCertiPHY
from waveforge6g.waveforms import create_waveform


if __name__=='__main__':
    path=OUT/'memory.json'
    if path.exists():raise RuntimeError('memory diagnostic preserved')
    budget=Budget();rows=[]
    try:
        for profile in 'ACE':
            channel=TDLProfile(profile,976000+ord(profile),delay_spread_s=300e-9,doppler_hz=5)
            wave=create_waveform('ofdm',1024,128,16)
            y=np.random.default_rng(76).normal(size=1024).astype(complex)
            for method in ['gray','deep','selective']:
                budget.check();tracemalloc.start()
                receiver=ProfileCertiPHY(wave,channel,10**(-2.8),'16qam')
                rule,refiner=configuration(dict(method=method,delta=.01,gate=[.01,12.]))
                result=receiver.solve(y,rule,refiner=refiner)
                current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();budget.calls+=1
                rows.append(dict(profile=profile,method=method,peak_traced_bytes=peak,current_traced_bytes=current,
                                 receiver_resident_estimate=result['array_storage_bytes'],nnz=receiver.nnz,
                                 note='Python/NumPy traced allocations, not OS RSS; tracing overhead excluded from primary CPU comparisons'))
        write(path,rows);write(DOC/'memory.json',rows)
    finally:budget.save()
