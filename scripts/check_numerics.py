"""New fractional-model diagnostics and tiny independent Decimal checks."""
import json
from pathlib import Path
import numpy as np
from scipy.special import j0
from waveforge6g.channels.tdl_profile import TDLProfile,fractional_kernel
from waveforge6g.receivers.certiphy_selective import ProfileCertiPHY,SelectiveRefiner
from waveforge6g.receivers.certiphy_deep import circulant_envelope
from waveforge6g.receivers.certiphy import StopRule,bit_margins
from waveforge6g.experiments.precision import decimal_reference
from waveforge6g.waveforms import create_waveform
from waveforge6g.core.modulation import demodulate
from run_study import OUT,DOC,write,Budget,reference


def run(budget):
    path=OUT/'diagnostics.json'
    if path.exists():raise RuntimeError('completed diagnostics preserved')
    responses=[]; f=np.linspace(-.5,.5,8193)
    for delay in [.01,.1,.37,.5,.9,3.,7.73]:
        d,h=fractional_kernel(delay)
        response=np.exp(-2j*np.pi*f[:,None]*d)@h
        exact=np.exp(-2j*np.pi*f*(delay+16))
        error=abs(response-exact); band=abs(f)<=.4
        responses.append(dict(delay=delay,energy=float(h@h),rms_full=float(np.sqrt(np.mean(error**2))),max_full=float(error.max()),
                              rms_80percent=float(np.sqrt(np.mean(error[band]**2))),max_80percent=float(error[band].max())))
    # Independent ensembles, not per-realization power normalization.
    draws=[]
    for seed in range(300):
        c=TDLProfile('C',960000+seed,doppler_hz=500)
        g=c.coefficients(np.array([0,100,500,1000]))/np.sqrt(c.power)[:,None]
        draws.append(g)
    draws=np.concatenate(draws)
    lag=np.array([0,100,500,1000]); correlation=np.mean(draws[:,0,None].conj()*draws,axis=0)
    doppler=dict(samples=len(draws),power=float(np.mean(abs(draws)**2)),lags=lag.tolist(),
                 measured_real=correlation.real.tolist(),measured_imag=correlation.imag.tolist(),
                 jakes=j0(2*np.pi*500*lag/7.68e6).tolist())
    precision=[]
    for waveform in ['ofdm','otfs','afdm']:
        for noise in [1e-2,1e-8]:
            wave=create_waveform(waveform,16,16,4)
            c=TDLProfile('C',962000,delay_spread_s=20e-9,doppler_hz=1500,half=4)
            rx=ProfileCertiPHY(wave,c,noise,'64qam')
            rng=np.random.default_rng(123); y=rng.normal(size=16)+1j*rng.normal(size=16)
            z80=decimal_reference(rx.matrix.toarray(),y,noise,80)
            z110=decimal_reference(rx.matrix.toarray(),y,noise,110)
            ref=wave.analysis(z110); rb=demodulate(ref,'64qam')
            result=rx.solve(y,StopRule(method='gray',delta=0),refiner=SelectiveRefiner())
            budget.calls+=1
            weights,info=circulant_envelope(rx.matrix,noise)
            defect=None
            if weights is not None:
                fmat=np.fft.fft(np.eye(16),axis=0,norm='ortho')
                matrix=(rx.adjoint@rx.matrix).toarray()+noise*np.eye(16)
                defect=float(np.linalg.eigvalsh(matrix-fmat.conj().T@np.diag(weights)@fmat).min())
            precision.append(dict(waveform=waveform,noise=noise,decimal80_110_error=float(np.linalg.norm(z80-z110)),
                                  disagreement=float(np.mean(result['bits']!=rb)),bound=result['disagreement_bound'],
                                  met=result['met_requested_bound'],min_margin=float(bit_margins(ref,'64qam').min()),
                                  envelope_defect_min=defect,fp_certified=False))
    # New high-Doppler N1024/64QAM stress, outside confirmation inference.
    wave=create_waveform('afdm',1024,128,16); c=TDLProfile('C',963000,doppler_hz=1800)
    rx=ProfileCertiPHY(wave,c,1e-3,'64qam'); rng=np.random.default_rng(12)
    y=rng.normal(size=1024)+1j*rng.normal(size=1024)
    ref,known,radius=reference(rx,y)
    result=rx.solve(y,StopRule(method='gray',delta=.01),refiner=SelectiveRefiner())
    budget.calls+=1
    stress=dict(n=1024,modulation='64qam',fd=1800,disagreement=float(np.mean(result['bits']!=demodulate(ref,'64qam'))),
                bound=result['disagreement_bound'],met=result['met_requested_bound'],status=result['status'],reference_unknown=int((~known).sum()),
                gate_enabled=result['refinement'].get('gate_enabled'),fp_certified=False)
    result=dict(fractional_response=responses,doppler=doppler,decimal=precision,stress=stress,
                historical_stress='Reuse tests/test_certiphy.py and test_certiphy_deep.py: boundary, cancellation, drift, invalid Ritz, exact Gray enumeration; no historical experiments rerun')
    write(path,result);write(DOC/'diagnostics.json',result)


if __name__=='__main__':
    budget=Budget()
    try:run(budget)
    finally:budget.save()
