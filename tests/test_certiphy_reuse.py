import numpy as np
import pytest
from dataclasses import replace
from waveforge6g.channels.reuse_paths import (synthetic_frame,tdl_frame,accurate_kernel,SparsePathAssembly,path_change_bound,OversampledWave)
from waveforge6g.receivers.certiphy_reuse import (ReuseCertiPHY,ReuseState,ReusePolicy,ReuseRefiner,ColdPathCertiPHY,ColdRefiner,channel_norm,direction_values)
from waveforge6g.receivers.certiphy import StopRule,bit_margins
from waveforge6g.core.modulation import modulate,demodulate
from waveforge6g.waveforms import create_waveform


@pytest.mark.parametrize('name',['ofdm','otfs','afdm'])
@pytest.mark.parametrize('kind',['integer','fractional'])
def test_full_propagation_and_csr_cache(name,kind):
    n=128;cp=96
    c=synthetic_frame(3,n,cp,4,'moderate') if kind=='integer' else tdl_frame('C',3,n,cp,4,40.)
    w=create_waveform(name,n,cp,16);a=SparsePathAssembly(w,c);matrix,cost=a.build(c)
    x=np.random.default_rng(4).normal(size=n)+1j*np.random.default_rng(5).normal(size=n)
    tx=np.r_[w.prefix_phases*x[-cp:],x]
    np.testing.assert_allclose(matrix@x,c.apply(tx)[cp:],rtol=1e-11,atol=1e-11)
    np.testing.assert_allclose(matrix.toarray(),c.matrix(w).toarray(),rtol=1e-11,atol=1e-11)
    assert cost>0 and a.static_work>0


@pytest.mark.parametrize('event',['phase','gain','delay','birth','death','frequency'])
def test_parameter_bound_dominates_actual_operator_change(event):
    n=32;cp=16;w=create_waveform('afdm',n,cp,8);old=synthetic_frame(11,n,cp,0,'slow');new=replace(old,start_index=83)
    a=[v.copy() for v in new.amplitudes];f=[v.copy() for v in new.frequencies];k=list(new.kernels)
    if event=='phase':a[1]*=np.exp(.02j)
    if event=='gain':a[2]*=.83
    if event=='frequency':f[2]*=31
    if event=='delay':k[2]=(np.array([6]),np.array([1.]))
    new=replace(new,amplitudes=tuple(a),frequencies=tuple(f),kernels=tuple(k))
    if event=='birth':new=replace(new,ids=new.ids+('b',),kernels=new.kernels+((np.array([3]),np.array([1.])),),amplitudes=new.amplitudes+(np.array([.3j]),),frequencies=new.frequencies+(np.array([12.]),))
    if event=='death':new=replace(new,ids=new.ids[:-1],kernels=new.kernels[:-1],amplitudes=new.amplitudes[:-1],frequencies=new.frequencies[:-1])
    eps,_=path_change_bound(old,new)
    actual=np.linalg.norm((new.matrix(w)-old.matrix(w)).toarray(),2)
    assert actual<=eps*(1+1e-10)+1e-12


@pytest.mark.parametrize('name',['ofdm','otfs','afdm'])
def test_oversampled_unitary_completion_and_directions(name):
    base=create_waveform(name,32,16,8);w=OversampledWave(base,64,32)
    rng=np.random.default_rng(4);s=rng.normal(size=64)+1j*rng.normal(size=64)
    np.testing.assert_allclose(w.analysis(w.synthesis(s)),s,atol=2e-13)
    x=w.synthesis(np.r_[s[:32],np.zeros(32)])
    spectrum=np.fft.fft(x,norm='ortho');assert np.linalg.norm(spectrum[w.unused])<1e-12
    m=1.+rng.random(64);basis=np.eye(64,dtype=complex)
    exact=np.array([np.sum(abs(np.fft.fft(w.synthesis(e),norm='ortho'))**2/m) for e in basis])
    np.testing.assert_allclose(direction_values(w,m),exact,atol=1e-12)


def test_full_occupied_band_and_group_delay():
    omega=np.linspace(-np.pi/2,np.pi/2,4097)
    for fraction in [0.,.01,.25,.5,.75,.99]:
        d,h=accurate_kernel(fraction)
        kernel=np.exp(-1j*omega[:,None]*d)
        response=kernel@h;ideal=np.exp(-1j*omega*(32+fraction))
        assert np.max(abs(response-ideal))<1e-4
        gd=-np.imag((kernel@(-1j*d*h))/response)
        assert np.max(abs(gd-(32+fraction)))<.002


def test_scaled_loewner_reuse_noise_updates_and_invalidation():
    w=create_waveform('ofdm',32,16,8);st=ReuseState(ReusePolicy(minimum_scale=.5))
    old=synthetic_frame(71,32,16,0,'slow')
    # Dominant invertible channel ensures the conditional envelope exists.
    old=replace(old,amplitudes=(np.array([1.]),np.array([.05]),np.array([.02]),np.array([.01])))
    for epoch,noise in [(0,.01),(1,.01),(2,.011),(3,.008)]:
        c=replace(old,start_index=epoch*48)
        rx=ReuseCertiPHY(w,c,noise,'qpsk',st);refiner=ReuseRefiner(st)
        from waveforge6g.receivers.certiphy import WorkLedger
        refiner.prepare(rx,WorkLedger(np.inf));assert refiner.weights is not None
        a=(rx.adjoint@rx.matrix).toarray()+noise*np.eye(32)
        ff=np.fft.fft(np.eye(32),axis=0,norm='ortho')
        lower=ff.conj().T@np.diag(refiner.weights)@ff
        assert np.linalg.eigvalsh(a-lower).min()>-1e-10
        if epoch==1:assert refiner.transition['action']=='update'
    c=replace(old,amplitudes=(np.array([-.01]),np.array([1.]),np.array([.5]),np.array([.4])))
    rx=ReuseCertiPHY(w,c,.001,'qpsk',st);refiner=ReuseRefiner(st);refiner.prepare(rx,WorkLedger(np.inf))
    assert refiner.transition['action']=='rebuild' and st.metrics['invalidations']>=1


def test_exact_reuse_lagged_spd_and_no_payload_warm_start():
    w=create_waveform('ofdm',32,16,8);st=ReuseState();c=synthetic_frame(71,32,16,0,'slow')
    c=replace(c,frequencies=tuple(np.zeros_like(f) for f in c.frequencies),amplitudes=(np.array([1.]),np.array([.05]),np.array([.02]),np.array([.01])))
    rng=np.random.default_rng(1);y=rng.normal(size=32)+1j*rng.normal(size=32)
    for i in range(2):
        rx=ReuseCertiPHY(w,replace(c,start_index=48*i),.01,'qpsk',st)
        r=rx.solve(y,StopRule(method='gray',delta=.01),refiner=ReuseRefiner(st))
        if i:assert r['reuse']['action']=='reuse' and not r['reuse']['jacobi_rebuilt']
        assert np.isclose(sum(r['work_parts'].values()),r['work'])
        ref=np.linalg.solve((rx.adjoint@rx.matrix).toarray()+.01*np.eye(32),rx.adjoint@y)
        mismatch=np.mean(r['bits']!=demodulate(w.analysis(ref),'qpsk'))
        assert mismatch<=r['disagreement_bound']+1e-12
        assert not r['floating_point_certified']


def test_frozen_gray_and_deep_kernel_files_unchanged():
    import hashlib
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    reference=json.loads((root/'tests/reference/receiver_numerics.json').read_text())
    for name in ['certiphy_deep.py','certiphy_selective.py']:
        p='src/waveforge6g/receivers/'+name
        assert hashlib.sha256((root/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==reference['kernels'][name]
    # R7 extends only solve's output mask and charged preparation hook. Existing
    # numerical/Gray functions and PCG normal/dual operators remain unchanged.
    import ast
    p='src/waveforge6g/receivers/certiphy.py'
    new=(root/p).read_text()
    def numerical(text):
        tree=ast.parse(text)
        # Normalize the AST field added in Python 3.12 for cross-version fixtures.
        for node in ast.walk(tree):
            if hasattr(node,'type_params'):
                node._fields=tuple(f for f in node._fields if f!='type_params')
        return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.walk(tree)
                if isinstance(n,ast.FunctionDef) and n.name in ['bit_margins','gray_energy_bound','targeted_gray_bound','physical_spectral_bounds','normal','_dual']}
    assert reference['functions']==numerical(new)


@pytest.mark.parametrize('modulation',['bpsk','qpsk','16qam','64qam'])
def test_cached_geometry_matches_frozen_bounds_and_ties(modulation):
    from waveforge6g.receivers.reuse_geometry import GrayGeometry
    from waveforge6g.core.modulation import bits_per_symbol
    from waveforge6g.receivers.certiphy import targeted_gray_bound
    width=bits_per_symbol(modulation);g=GrayGeometry(width);rng=np.random.default_rng(width)
    for _ in range(10):
        s=rng.normal(size=32)+1j*rng.normal(size=32)
        if width>1:s[:len(g.cuts)]=g.cuts+1j*g.cuts
        m=bit_margins(s,modulation);np.testing.assert_array_equal(g.margins(s,modulation),m)
        active=rng.random((32,width))<.7;target=min(4,int(active.sum())-1)
        for radius in [.001,.1,1.,10.]:
            assert g.targeted(s,modulation,radius,active,target,m)==targeted_gray_bound(s,modulation,radius,active,target,m)


def test_oversampled_ofdm_gray_uses_symbol_order_weights():
    from waveforge6g.receivers.certiphy_deep import region_costs
    from waveforge6g.receivers.certiphy import WorkLedger
    base=create_waveform('ofdm',16,8,8);w=OversampledWave(base,32,16)
    c=synthetic_frame(2,32,16,0,'slow');rx=ColdPathCertiPHY(w,c,.01,'16qam')
    refiner=ColdRefiner('spectral');refiner.receiver=rx;refiner.weights=np.arange(1,33,dtype=float);refiner.current_eta=.005
    refiner.inverse_directions=direction_values(w,refiner.weights)
    centers=np.full(32,.08+.08j);active=np.ones((32,4),bool);margins=bit_margins(centers,'16qam')
    # Capture the actual region weights at the existing Deep kernel boundary.
    import waveforge6g.receivers.certiphy_deep as module
    original=module.region_costs;captured=[]
    def capture(s,mod,a,weights=None):captured.append(weights.copy());return original(s,mod,a,weights)
    from unittest.mock import patch
    with patch.object(module,'region_costs',capture):
        refiner.bound(centers,active,1,1.,1.,0.,WorkLedger(np.inf),margins)
    assert captured
    expected=refiner.weights[np.r_[w.occupied,w.unused]]
    np.testing.assert_array_equal(captured[0],np.r_[expected,expected])


def test_current_csi_snapshots_own_immutable_backing():
    c=synthetic_frame(2,32,16,0,'slow')
    for a in [*c.amplitudes,*c.frequencies,*[h for d,h in c.kernels]]:
        with pytest.raises(ValueError):a.setflags(write=True)


def test_multitone_fractional_delay_change_bound_with_cancellation():
    n=32;cp=16;w=create_waveform('afdm',n,cp,8)
    old=synthetic_frame(11,n,cp,0,'slow')
    amplitudes=tuple(np.array([a[0],-.7*a[0],.3j*a[0]]) for a in old.amplitudes)
    frequencies=tuple(np.array([2.,3.,-5.]) for _ in old.ids)
    kernels=tuple(accurate_kernel(float(d[0])+.25,half=3) for d,h in old.kernels)
    old=replace(old,amplitudes=amplitudes,frequencies=frequencies,kernels=kernels)
    new=replace(old,start_index=48,amplitudes=tuple(a*np.exp(.01j) for a in amplitudes),
        frequencies=tuple(f+np.array([.1,-.2,.05]) for f in frequencies),
        kernels=tuple(accurate_kernel(float(d[0])+.27,half=3) for d,h in kernels))
    eps,_=path_change_bound(old,new)
    assert np.linalg.norm((new.matrix(w)-old.matrix(w)).toarray(),2)<=eps


def test_budget_capped_ofdm_direction_alignment_is_conservative():
    from waveforge6g.receivers.certiphy import WorkLedger
    w=OversampledWave(create_waveform('ofdm',16,8,8),32,16)
    rx=ColdPathCertiPHY(w,synthetic_frame(2,32,16,0,'slow'),.01,'16qam')
    refiner=ColdRefiner('spectral');refiner.receiver=rx
    refiner.weights=np.geomspace(.02,4.,32);native=refiner.weights.copy()
    residual=np.full(32,1e-6+1e-6j)
    # No room for even the alignment fallback: preserve valid original radius.
    ledger=WorkLedger(1.)
    np.testing.assert_array_equal(refiner.radii(1.,.3,0.,ledger,residual),np.full(32,.3))
    assert refiner.current_eta is None and refiner.inverse_directions is None
    np.testing.assert_array_equal(refiner.weights,native)
    # Sufficient budget uses the worst direction, and restores the cache.
    ledger=WorkLedger(np.inf)
    radii=refiner.radii(1.,.3,0.,ledger,residual)
    assert np.all(radii<=.3) and np.all(radii>=0)
    assert ledger.parts['payload_direction_fallback']==128
    assert refiner.inverse_directions is None
    np.testing.assert_array_equal(refiner.weights,native)


def test_payload_does_not_influence_cache_reuse_transition():
    w=create_waveform('ofdm',32,16,8)
    histories=[]
    for payload_seed in [1,123]:
        st=ReuseState();transitions=[]
        rng=np.random.default_rng(payload_seed)
        for epoch in range(3):
            c=synthetic_frame(71,32,16,epoch,'slow')
            c=replace(c,amplitudes=(np.array([1.]),np.array([.05]),np.array([.02]),np.array([.01])))
            rx=ReuseCertiPHY(w,c,.01,'qpsk',st)
            refiner=ReuseRefiner(st)
            y=rng.normal(size=32)+1j*rng.normal(size=32)
            rx.solve(y,StopRule(method='gray',delta=.01,max_iterations=8),refiner=refiner)
            transitions.append((dict(refiner.transition),dict(st.metrics),st.anchor_weights.copy()))
        histories.append(transitions)
        assert not any(k in vars(st) for k in ['bits','reference','truth','solution','residual','received'])
    for a,b in zip(*histories):
        assert a[0]==b[0] and a[1]==b[1]
        np.testing.assert_array_equal(a[2],b[2])


def test_global_phase_can_invalidate_a_bound_without_changing_gram():
    from waveforge6g.receivers.certiphy_deep import circulant_envelope
    w=create_waveform('ofdm',32,16,8)
    old=synthetic_frame(71,32,16,0,'slow')
    old=replace(old,amplitudes=(np.array([1.]),np.array([.05]),np.array([.02]),np.array([.01])))
    new=replace(old,amplitudes=tuple(-a for a in old.amplitudes))
    a=old.matrix(w);b=new.matrix(w)
    np.testing.assert_allclose((a.conj().T@a).toarray(),(b.conj().T@b).toarray(),atol=1e-13)
    eps,_=path_change_bound(old,new);weights,_=circulant_envelope(a,.01)
    assert weights is not None
    assert np.min(weights-2*channel_norm(old)*eps)<0


def test_noncanonical_duplicate_offsets_rejected_before_unsafe_reuse():
    c=synthetic_frame(1,32,16,0,'slow');kernels=list(c.kernels)
    # Previously dictionary/mixing assignment lost the first contribution.
    # A physical duplicate must be coalesced into one coefficient by the caller.
    for h in [np.array([1.,1.]),np.array([100.,1.])]:
        kernels[0]=(np.array([0,0]),h)
        with pytest.raises(ValueError,match='unique delay offsets'):
            replace(c,kernels=tuple(kernels))


@pytest.mark.parametrize('changed',[0,1])
def test_operator_bound_keeps_distinct_native_identity_contributions(changed):
    w=create_waveform('ofdm',32,16,8)
    old=replace(synthetic_frame(1,32,16,0,'slow'),ids=(0,'0',2,3))
    amplitudes=list(old.amplitudes);amplitudes[changed]=100*amplitudes[changed]
    new=replace(old,amplitudes=tuple(amplitudes));eps,parts=path_change_bound(old,new)
    assert len(parts)==len(old.ids)
    assert np.linalg.norm((new.matrix(w)-old.matrix(w)).toarray(),2)<=eps
