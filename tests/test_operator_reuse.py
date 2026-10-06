from dataclasses import replace
from decimal import Decimal, localcontext
import numpy as np
import pytest
from scipy.sparse import csr_matrix, coo_matrix
from scipy.linalg import eigvalsh
from waveforge6g.receivers.operator_reuse import (relative_scale, aligned_change,
    sparse_norm_upper, ValidatorPolicy, OperatorReuseValidator, ValidatedRefiner,
    ScalarEnvelopePreparation, PayloadReuseRefiner)
from waveforge6g.receivers.certiphy_reuse import ReuseState,ReusePolicy,ReuseCertiPHY
from waveforge6g.receivers.certiphy import StopRule,WorkLedger
from waveforge6g.receivers.payload_selection import PayloadSelection
from waveforge6g.channels.reuse_paths import synthetic_frame,OversampledWave,path_change_bound
from waveforge6g.waveforms import create_waveform
from waveforge6g.core.modulation import demodulate


@pytest.mark.parametrize('noise',[1e-240,1e-6,1.,1e240])
@pytest.mark.parametrize('ratio',[.01,1.,100.])
def test_relative_formula_decimal_and_extreme_scale(noise,ratio):
    e=np.sqrt(noise)*.3;b=noise*ratio
    with localcontext() as ctx:
        ctx.prec=90
        a,bb,ee=map(Decimal.from_float,[noise,b,e])
        s=a+bb+ee*ee
        d=((a-bb)**2+2*ee*ee*(a+bb)+ee**4).sqrt()
        exact=2*bb/(s+d)
    g=relative_scale(e,noise,b)
    assert g<=float(exact)*(1+1e-15)
    assert np.isclose(g,float(exact),rtol=1e-12,atol=0)
    assert relative_scale(0,noise,b)==min(1.,ratio)


@pytest.mark.parametrize('shape',[(7,4),(4,7),(6,6)])
def test_complex_rectangular_relative_loewner_and_tight_scalar(shape):
    rng=np.random.default_rng(9)
    c=rng.normal(size=shape)+1j*rng.normal(size=shape)
    e=.02*(rng.normal(size=shape)+1j*rng.normal(size=shape))
    phase=np.exp(1j*rng.normal(size=shape[0]))
    t=phase[:,None]*c+e
    eps=np.linalg.norm(e,2);a=.01;b=.04
    g=relative_scale(eps,a,b)
    old=c.conj().T@c+a*np.eye(shape[1]);new=t.conj().T@t+b*np.eye(shape[1])
    assert eigvalsh(new-g*old).min()>-1e-12
    exact=relative_scale(.3,a,b);ca=.3/(1-exact)
    assert np.isclose(((ca-.3)**2+b)/(ca*ca+a),exact,rtol=1e-12)


@pytest.mark.parametrize('mode',['common','row'])
def test_phase_invariance_not_allclose_and_sparse_union(mode):
    rng=np.random.default_rng(11);a=csr_matrix(rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)))
    for phase in [-1.,1j,np.exp(.7j)]:
        t=a*phase
        d=aligned_change(t,a,mode)
        assert d['epsilon']<1e-11
        if phase in (-1.,1j):assert d['epsilon']==0 and relative_scale(d['epsilon'],.01,.01)==1
    phase=np.exp(1j*np.arange(8)*.2)
    t=a.multiply(phase[:,None]).tocsr()
    d=aligned_change(t,a,'row');assert d['epsilon']<1e-11
    assert aligned_change(t,a,'common')['epsilon']>1
    # Disjoint supports, zero rows, duplicates and exact path cancellation.
    aa=coo_matrix(([1.,-1.,2.],([0,0,2],[1,1,3])),shape=(8,8)).tocsr()
    tt=coo_matrix(([.3,2.],([1,2],[0,3])),shape=(8,8)).tocsr()
    d=aligned_change(tt,aa,'row')
    assert d['epsilon']>=np.linalg.norm((tt-aa).toarray(),2)-1e-12
    assert aligned_change(csr_matrix((8,8)),csr_matrix((8,8)))['epsilon']==0


def model(name='ofdm',oversampled=False):
    w=OversampledWave(create_waveform(name,16,8,8),32,16) if oversampled else create_waveform(name,32,16,8)
    c=synthetic_frame(4,32,16,0,'slow')
    c=replace(c,amplitudes=(np.array([1.]),np.array([.05]),np.array([.02]),np.array([.01])))
    return w,c


@pytest.mark.parametrize('name',['ofdm','otfs','afdm'])
@pytest.mark.parametrize('method',['gray','radau','stable_fast'])
def test_noncontinuous_payload_and_partial_gray_bits_same_full_reference(name,method):
    w,c=model(name,True);state=ReuseState(ReusePolicy(diagonal_age=1))
    rx=ReuseCertiPHY(w,c,.02,'16qam',state)
    rng=np.random.default_rng(17);y=rng.normal(size=32)+1j*rng.normal(size=32)
    mask=np.zeros((32,4),bool);mask[[0,3,7,15],[1,3,0,2]]=True
    result=rx.solve(y,StopRule(method=method,delta=0,max_iterations=128,period=2),output_bit_mask=mask,keep_trace=True)
    assert len(result['bits'])==4 and result['payload_bit_count']==4
    assert result['certified'].shape==(4,)
    z=np.linalg.solve((rx.adjoint@rx.matrix).toarray()+rx.noise*np.eye(32),rx.adjoint@y)
    reference=demodulate(w.analysis(z),'16qam').reshape(32,4)[mask]
    if method!='stable_fast':
        assert np.mean(reference!=result['bits'])<=result['disagreement_bound']
        assert result['met_requested_bound']
    assert all(len(t['output'])==4 for t in result['trace'])
    assert result['floating_point_certified'] is False


@pytest.mark.parametrize('name',['ofdm','otfs','afdm'])
def test_selected_weighted_refinement_and_full_energy(name):
    w,c=model(name,True);st=ReuseState(ReusePolicy(diagonal_age=1))
    rx=ReuseCertiPHY(w,c,.01,'16qam',st)
    rng=np.random.default_rng(6);y=rng.normal(size=32)+1j*rng.normal(size=32)
    validator=OperatorReuseValidator(st)
    r=rx.solve(y,StopRule(method='gray',delta=.05,period=2),payload_indices=np.arange(16),refiner=ValidatedRefiner(validator),keep_trace=True)
    assert len(r['bits'])==64 and len(r['soft'])==32 and len(r['time_estimate'])==32
    ref=np.linalg.solve((rx.adjoint@rx.matrix).toarray()+.01*np.eye(32),rx.adjoint@y)
    true_bits=demodulate(w.analysis(ref)[:16],'16qam')
    assert np.mean(r['bits']!=true_bits)<=r['disagreement_bound']+1e-12
    assert np.isclose(sum(r['work_parts'].values()),r['work'])


def test_anchor_drift_noise_and_actual_rhs_no_history_leak():
    w,c=model();st=ReuseState(ReusePolicy(diagonal_age=1));rng=np.random.default_rng(8)
    anchor=None
    for epoch in range(7):
        frame=replace(c,amplitudes=tuple(a*np.exp(1j*.3*epoch) for a in c.amplitudes))
        rx=ReuseCertiPHY(w,frame,.01 if epoch<5 else .02,'qpsk',st)
        v=OperatorReuseValidator(st);y=rng.normal(size=32)+1j*rng.normal(size=32)
        r=rx.solve(y,StopRule(method='radau',delta=.01),preparer=ScalarEnvelopePreparation(v))
        if epoch==0:anchor=st.anchor_matrix.copy()
        np.testing.assert_array_equal(st.anchor_matrix.toarray(),anchor.toarray())
        if epoch==1:assert v.transition['action']=='update' and v.transition['scale']>.999999
        a=(rx.adjoint@rx.matrix).toarray()+rx.noise*np.eye(32)
        f=np.fft.fft(np.eye(32),axis=0,norm='ortho')
        assert eigvalsh(a-f.conj().T@np.diag(v.weights)@f).min()>-1e-10
        ref=np.linalg.solve(a,rx.adjoint@y)
        assert np.mean(r['bits']!=demodulate(w.analysis(ref),'qpsk'))<=r['disagreement_bound']+1e-12
        assert not any(k in vars(st) for k in ['solution','received','residual','truth','reference'])


def test_invalid_masks_and_no_positive_underflow_fabrication():
    for kwargs in [dict(payload_indices=[1.,2.]),dict(payload_indices=[1,1]),dict(output_bit_mask=np.zeros((4,2),bool))]:
        with pytest.raises(ValueError):PayloadSelection(4,2,**kwargs)
    assert relative_scale(1e300,1e-300,1e-300)==0
    for args in [(1,0,1),(-1,1,1),(np.inf,1,1)]:
        with pytest.raises(ValueError):relative_scale(*args)


@pytest.mark.parametrize('condition',[1.,1e4,1e8])
@pytest.mark.parametrize('noise',[1e-6,.01])
def test_ill_conditioned_parallel_and_transverse_changes(condition,noise):
    a=np.diag([1.,.1,1/condition]).astype(complex)
    for direction in [np.diag([1.,0.,0.]),np.diag([0.,0.,1.]),np.roll(np.eye(3),1,axis=0)]:
        e=.002j*direction;t=np.exp(.3j)*a+e
        detail=aligned_change(csr_matrix(t),csr_matrix(a),'row')
        g=relative_scale(detail['epsilon'],noise,noise)
        old=a.conj().T@a+noise*np.eye(3);new=t.conj().T@t+noise*np.eye(3)
        assert eigvalsh(new-g*old).min()>=-1e-12


def test_matrix_reordering_is_measured_or_coordinate_key_rebuilt():
    w,c=model();a=c.matrix(w);p=np.arange(32)[::-1]
    d=aligned_change(a[:,p],a,'row')
    assert d['epsilon']>0
    old=(a.conj().T@a).toarray()+.01*np.eye(32)
    new=(a[:,p].conj().T@a[:,p]).toarray()+.01*np.eye(32)
    assert eigvalsh(new-relative_scale(d['epsilon'],.01,.01)*old).min()>-1e-10
    st=ReuseState();rx=ReuseCertiPHY(w,c,.01,'qpsk',st)
    v=OperatorReuseValidator(st);v.prepare(rx,WorkLedger(np.inf));assert st.anchor is not None
    # Prefix change is a coordinate/model-key change, despite the same N.
    w2=create_waveform('ofdm',32,12,8);c2=replace(c,cp=12)
    ReuseCertiPHY(w2,c2,.01,'qpsk',st)
    assert st.anchor is None


def test_cancelled_paths_do_not_create_an_envelope_or_fake_boundary_certificate():
    w,c=model();kernels=((np.array([0]),np.array([1.])),)*2
    c=replace(c,ids=(0,'0'),kernels=kernels,amplitudes=(np.array([1.]),np.array([-1.])),frequencies=(np.array([0.]),)*2)
    st=ReuseState(ReusePolicy(diagonal_age=1))
    for epoch in range(2):
        rx=ReuseCertiPHY(w,replace(c,start_index=epoch*48),.01,'qpsk',st)
        validator=OperatorReuseValidator(st)
        r=rx.solve(np.ones(32),StopRule(method='gray',delta=0,max_iterations=4),payload_indices=np.array([1,3,9]),refiner=ValidatedRefiner(validator))
        assert validator.weights is None and not r['met_requested_bound']
        assert not r['certified'].any() and r['unknown_fraction']==1
        if epoch:assert validator.transition['reason']=='cached_no_envelope'


def test_payload_order_and_integer_granularity_are_explicit():
    selection=PayloadSelection(8,4,payload_indices=np.array([7,0,3]))
    np.testing.assert_array_equal(selection.symbols,[0,3,7])
    mask=np.zeros((8,4),bool);mask[0]=True;mask[3,:3]=True
    selection=PayloadSelection(8,4,output_bit_mask=mask)
    assert selection.count==7
    assert int(np.floor(.01*selection.count))==0


def test_validator_budget_exhaustion_keeps_anchor_and_returns_no_free_envelope():
    w,c=model();st=ReuseState(ReusePolicy(diagonal_age=1))
    rx=ReuseCertiPHY(w,c,.01,'qpsk',st)
    old=OperatorReuseValidator(st);paid=WorkLedger(np.inf);old.prepare(rx,paid)
    assert old.weights is not None and paid.parts['immutable_operator_anchor']>0
    anchor=st.anchor_matrix.copy();weights=st.anchor_weights.copy()
    changed=replace(c,amplitudes=tuple(a*np.exp(.3j) for a in c.amplitudes))
    rx=ReuseCertiPHY(w,changed,.01,'qpsk',st)
    equality_fee=4*(rx.nnz+st.anchor_matrix.nnz)+8*rx.n
    limited=WorkLedger(equality_fee);v=OperatorReuseValidator(st);v.prepare(rx,limited)
    assert v.transition['reason']=='alignment_budget'
    assert v.transition['action']=='fallback' and v.weights is None
    assert limited.total==equality_fee
    np.testing.assert_array_equal(st.anchor_matrix.toarray(),anchor.toarray())
    np.testing.assert_array_equal(st.anchor_weights,weights)
