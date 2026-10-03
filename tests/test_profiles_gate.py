import numpy as np
import pytest
from waveforge6g.channels.tdl_profile import TDLProfile, fractional_kernel
from waveforge6g.channels.doubly_selective import ChannelRealization
from waveforge6g.receivers.budgeted import time_channel
from waveforge6g.receivers.certiphy import StopRule
from waveforge6g.receivers.certiphy_selective import ProfileCertiPHY, SelectiveRefiner
from waveforge6g.receivers.certiphy_deep import DeepRefiner
from waveforge6g.waveforms import create_waveform


@pytest.mark.parametrize('name', ['ofdm','otfs','afdm'])
def test_full_fractional_propagation(name):
    w=create_waveform(name,256,128,16)
    c=TDLProfile('C',51,delay_spread_s=300e-9,doppler_hz=1300)
    rng=np.random.default_rng(8); x=rng.normal(size=256)+1j*rng.normal(size=256)
    np.testing.assert_allclose(c.apply(w.add_prefix(x),77)[128:],c.matrix(w,77)@x,atol=2e-14)


def test_integer_limit_and_causal_latency():
    d,h=fractional_kernel(3.,16)
    assert d.tolist()==[19] and h.tolist()==[1.]
    c=TDLProfile('A',8,half=0,delay_spread_s=0,doppler_hz=0)
    w=create_waveform('afdm',32,16,8)
    gains=c.coefficients([0])[:,0]
    old=ChannelRealization(np.zeros(c.n_paths,dtype=int),gains,np.zeros(c.n_paths),c.sample_rate_hz)
    np.testing.assert_allclose(c.matrix(w).toarray(),time_channel(w,old).toarray(),atol=1e-14)


def test_fractional_energy_and_no_rounding():
    d,h=fractional_kernel(.37)
    assert len(h)==33 and abs(np.sum(h*h)-1)<1e-15
    assert not np.allclose(h,np.eye(1,33,16).ravel())


def test_disabled_gate_exact_gray_and_charged():
    w=create_waveform('ofdm',128,64,16); c=TDLProfile('E',3,doppler_hz=1000)
    rx=ProfileCertiPHY(w,c,.02,'16qam'); y=np.random.default_rng(1).normal(size=128).astype(complex)
    rule=StopRule(method='gray',delta=.01)
    a=rx.solve(y,rule); b=rx.solve(y,rule,refiner=SelectiveRefiner(doppler_limit=0))
    np.testing.assert_array_equal(a['bits'],b['bits'])
    assert a['iterations']==b['iterations'] and b['work']==a['work']+80
    assert a['disagreement_bound']==b['disagreement_bound']
    assert not b['floating_point_certified']


def test_fractional_spectral_bounds_and_guard():
    c=TDLProfile('E',21,half=4,doppler_hz=900)
    w=create_waveform('ofdm',32,32,8); rx=ProfileCertiPHY(w,c,.001,'qpsk')
    a=(rx.adjoint@rx.matrix).toarray()+rx.noise*np.eye(32)
    ev=np.linalg.eigvalsh(a)
    assert rx.alpha <= ev[0] and rx.beta>=ev[-1]
    with pytest.raises(ValueError): c.matrix(create_waveform('ofdm',32,0,8))


def test_gate_has_only_available_inputs():
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(SelectiveRefiner.prepare)))
    attributes={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)}
    assert not attributes.intersection({'truth','reference','future','disagreement','ber'})


def test_enabled_gate_preserves_deep_semantics():
    w=create_waveform('ofdm',128,64,16);c=TDLProfile('E',82,doppler_hz=5)
    rx=ProfileCertiPHY(w,c,1e-8,'16qam');y=np.random.default_rng(19).normal(size=128).astype(complex)
    rule=StopRule(method='gray',delta=.01)
    a=rx.solve(y,rule,refiner=DeepRefiner('spectral'))
    b=rx.solve(y,rule,refiner=SelectiveRefiner())
    assert b['refinement']['gate_enabled']
    np.testing.assert_array_equal(a['bits'],b['bits'])
    assert a['iterations']==b['iterations'] and b['work']==a['work']+80
    assert a['disagreement_bound']==b['disagreement_bound']


def test_official_e_ricean_split_not_double_counted():
    c=TDLProfile('E',9)
    assert len(c.power)==15 and sum(c.los)==1
    assert c.delays[0]==c.delays[1]==0
    assert abs(10*np.log10(c.power[0]/c.power[1])-22)<1e-12
    assert abs(c.power.sum()-1)<1e-15
