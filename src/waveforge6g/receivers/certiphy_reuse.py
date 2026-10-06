"""R6 state reuse. R4/R4-Deep decision bounds and PCG solve are unchanged.

The transition uses current CSI and a deterministic path-operator perturbation
bound. No payload warm start, reference answer or future CSI is accepted.
"""
from dataclasses import dataclass,replace
import time
import numpy as np
from .certiphy import CertiPHY
from .certiphy_selective import ProfileCertiPHY
from .certiphy_deep import DeepRefiner,circulant_envelope,inverse_directions
from .budgeted import transform_work
from ..channels.reuse_paths import SparsePathAssembly,path_change_bound,accurate_kernel
from ..core.modulation import bits_per_symbol
from .reuse_geometry import GrayGeometry


@dataclass(frozen=True)
class ReusePolicy:
    minimum_scale: float=.5
    failed_envelope_age: int=10
    diagonal_age: int=20
    diagonal_variation: float=.05


def channel_norm(frame):
    return float(sum(abs(a).sum()*abs(h).sum() for a,(d,h) in zip(frame.amplitudes,frame.kernels)))


def direction_values(wave,weights):
    if hasattr(wave,'base'):
        return np.r_[inverse_directions(wave.base,weights[wave.occupied]),1/weights[wave.unused]+128*np.finfo(float).eps*(1+1/weights.min())]
    return inverse_directions(wave,weights)


def wave_work(wave):
    if not hasattr(wave,'base'):return transform_work(wave)
    return float(5*wave.n_symbols*np.log2(wave.n_symbols)+5*wave.payload_symbols*np.log2(wave.payload_symbols)+transform_work(wave.base)+4*wave.n_symbols)


def wave_static_work(wave):
    n=wave.n_symbols
    return float(96*n+48*wave.cp_length if wave.name=='afdm' else 8*n+4*wave.cp_length)


class ReuseState:
    def __init__(self,policy=None):
        self.policy=policy or ReusePolicy()
        self.assembly=None;self.diagonal=None;self.diagonal_frame=None;self.diagonal_noise=None;self.diagonal_last=-1
        self.anchor=None;self.anchor_weights=None;self.anchor_directions=None;self.anchor_norm=0.;self.anchor_noise=0.;self.anchor_last=-1
        self.wave_key=None;self.frame_index=-1;self.geometry=None
        self.metrics=dict(rebuilds=0,updates=0,reuses=0,invalidations=0,weak_updates=0,false_reuse_attempts=0,fallbacks=0)


class ReuseCertiPHY(CertiPHY):
    """Fresh full operator, persistent assembly and fixed-within-frame Jacobi.

    All warm method families, including Radau/residual, receive the same lagged
    Jacobi policy. A positive old diagonal remains a legal SPD preconditioner;
    mu is recomputed with the current alpha and that actual old maximum.
    """
    def __init__(self,wave,frame,noise,modulation,state=None):
        if noise<=0 or not np.isfinite(noise):raise ValueError('positive finite N0 required')
        start,cpu=time.perf_counter(),time.process_time()
        self.state=state or ReuseState();st=self.state;st.frame_index+=1;k=st.frame_index
        self.wave,self.channel,self.noise,self.modulation=wave,frame,float(noise),modulation
        self.n,self.width=wave.n_symbols,bits_per_symbol(modulation)
        signature=frame.signature();parts={}
        base=getattr(wave,'base',wave)
        wk=(wave.name,self.n,wave.cp_length,getattr(base,'c1',None),getattr(base,'c2',None),getattr(base,'subcarriers',None),getattr(base,'time_slots',None),self.width,hasattr(wave,'base'))
        if st.wave_key!=wk:
            st.assembly=None;st.diagonal=None;st.anchor=None;st.anchor_weights=None;st.anchor_directions=None
            st.wave_key=wk;parts['wave_geometry_static']=wave_static_work(wave)+16*2**self.width
            st.geometry=GrayGeometry(self.width)
        if st.assembly is None or st.assembly.signature!=signature:
            st.assembly=SparsePathAssembly(wave,frame);parts['operator_static']=st.assembly.static_work+64*sum(len(d) for d,h in frame.kernels)
            st.diagonal=None
        self.matrix,dynamic=st.assembly.build(frame);parts['operator_current']=dynamic
        self.adjoint=self.matrix.conj().T.tocsr();parts['adjoint_current']=16*self.matrix.nnz
        refresh=st.diagonal is None
        eps=0.
        if not refresh:
            eps,_=path_change_bound(st.diagonal_frame,frame)
            parts['diagonal_validation']=float(200*sum(len(f) for f in frame.frequencies)+16*sum(len(d) for d,h in frame.kernels))
            refresh=(k-st.diagonal_last>=st.policy.diagonal_age or eps>st.policy.diagonal_variation*max(channel_norm(st.diagonal_frame),1e-30)
                     or abs(noise-st.diagonal_noise)>.05*max(st.diagonal_noise,1e-30))
        if refresh:
            st.diagonal=np.asarray(abs(self.matrix).power(2).sum(axis=0)).ravel()+noise
            st.diagonal_frame=frame;st.diagonal_noise=noise;st.diagonal_last=k
            parts['jacobi_rebuild']=8*self.matrix.nnz+4*self.n
        else:parts['jacobi_reuse_check']=4*self.n
        self.diagonal=st.diagonal
        # Current certified spectrum does not inherit an old minimum Ritz value.
        self.alpha=float(noise);self.beta=float(noise+channel_norm(frame)**2)
        self.mu=(1-1e-12)*self.alpha/float(self.diagonal.max())
        self.nnz=self.matrix.nnz;self.transform=wave_work(wave)
        self.step_work=16*self.nnz+40*self.n+8;self.normal_work=16*self.nnz+12*self.n
        self.decode_work=4*self.n*self.width
        self.preparation_parts=parts;self.setup_work=float(sum(parts.values()))
        self.setup_elapsed=time.perf_counter()-start;self.setup_cpu=time.process_time()-cpu
        self.reuse_info=dict(jacobi_rebuilt=refresh,diagonal_change_bound=eps,static_rebuilt='operator_static' in parts)

    def solve(self,*args,**kwargs):
        result=self.state.geometry.solve(self,*args,**kwargs)
        result['work_parts'].pop('preparation',None);result['work_parts'].update(self.preparation_parts)
        assert np.isclose(sum(result['work_parts'].values()),result['work'],rtol=1e-13)
        result['reuse']={**self.reuse_info,**({} if kwargs.get('refiner') is None else kwargs['refiner'].transition)}
        result['cache_metrics']=dict(self.state.metrics)
        return result


class ColdPathCertiPHY(ProfileCertiPHY):
    """R4/R5 cold path for the new channel; all preparation paid every frame."""
    def __init__(self,wave,frame,noise,modulation):
        started,cpu=time.perf_counter(),time.process_time()
        if frame.filter_parameters:frame=replace(frame,kernels=tuple(accurate_kernel(d,half) for d,half in frame.filter_parameters))
        super().__init__(wave,frame,noise,modulation)
        self.setup_elapsed=time.perf_counter()-started;self.setup_cpu=time.process_time()-cpu
        self.transform=wave_work(wave)
        self.setup_work+=wave_static_work(wave)+16*2**self.width
        # Same observable safe spectrum as the warm families.
        self.alpha=float(noise);self.beta=float(noise+channel_norm(frame)**2)
        self.mu=(1-1e-12)*self.alpha/float(self.diagonal.max())


class PayloadRegionAlignment:
    def radii(self,energy,radius,guard,ledger,residual):
        if (self.weights is not None and self.inverse_directions is None
                and hasattr(self.receiver.wave,'base') and self.receiver.wave.name=='ofdm'):
            # A capped preparation may leave no symbol-order directions. Never
            # substitute native-order OFDM weights for the permuted completion.
            if not ledger.fits(4*self.receiver.n):
                self.current_eta=None
                return np.full(self.receiver.n,radius)
            ledger.add('payload_direction_fallback',4*self.receiver.n)
            self.inverse_directions=np.full(self.receiver.n,1/self.weights.min())
            try:return DeepRefiner.radii(self,energy,radius,guard,ledger,residual)
            finally:self.inverse_directions=None
        return DeepRefiner.radii(self,energy,radius,guard,ledger,residual)

    def bound(self,symbols,active,target,radius,energy,guard,ledger,margins):
        # The oversampled OFDM unitary completion permutes Fourier columns.
        # Residual energy uses native Fourier order, while Gray axis costs must
        # use actual symbol order. These are not interchangeable arrays.
        if self.weights is not None and hasattr(self.receiver.wave,'base') and self.receiver.wave.name=='ofdm':
            fee=4*self.receiver.n
            if not ledger.fits(fee):return int(active.sum())
            ledger.add('payload_region_permutation',fee)
            w=self.weights;wave=self.receiver.wave
            self.weights=w[np.r_[wave.occupied,wave.unused]]
            try:return DeepRefiner.bound(self,symbols,active,target,radius,energy,guard,ledger,margins)
            finally:self.weights=w
        return DeepRefiner.bound(self,symbols,active,target,radius,energy,guard,ledger,margins)


class ColdRefiner(PayloadRegionAlignment,DeepRefiner):
    def prepare(self,rx,ledger):
        self.receiver=rx;start=time.perf_counter()
        cost=100*rx.nnz+5*rx.n*np.log2(rx.n)+80*rx.n
        if ledger.fits(cost):
            ledger.add('deep_envelope_preparation',cost)
            self.weights,details=circulant_envelope(rx.matrix,rx.noise);self.metrics.update(details)
            if self.weights is not None:
                fee=20*rx.n*np.log2(rx.n)+40*rx.n
                if ledger.fits(fee):
                    ledger.add('deep_inverse_directions',fee)
                    self.inverse_directions=direction_values(rx.wave,self.weights)
        self.metrics['preparation_seconds']+=time.perf_counter()-start


class ReuseRefiner(PayloadRegionAlignment,DeepRefiner):
    def __init__(self,state):
        super().__init__('spectral');self.state=state;self.transition={}

    def prepare(self,rx,ledger):
        self.receiver=rx;st=self.state;start=time.perf_counter();k=st.frame_index
        p=st.policy
        # A scalar old A-energy radius is NEVER reused: residual/Radau are fresh.
        action='rebuild';eps=eta=scale=0.;reason='cold'
        validation=200*sum(len(f) for f in rx.channel.frequencies)+16*sum(len(d) for d,h in rx.channel.kernels)+12*rx.n
        if st.anchor is not None:
            if not ledger.fits(validation):
                self.transition=dict(action='fallback',reason='validation_budget');st.metrics['fallbacks']+=1;return
            ledger.add('reuse_validity',validation)
            eps,contributions=path_change_bound(st.anchor,rx.channel)
            eta=2*st.anchor_norm*eps+eps*eps+abs(rx.noise-st.anchor_noise)
            same=st.anchor.coefficient_key()==rx.channel.coefficient_key() and rx.noise==st.anchor_noise
            if st.anchor_weights is not None:
                # Loewner lower bound may omit PSD Delta_C^H Delta_C. Signed N0
                # update is tighter than taking abs(dN0) in an operator-norm bound.
                shift=2*st.anchor_norm*eps-(rx.noise-st.anchor_noise)
                proposal=st.anchor_weights-shift
                scale=float(np.min(proposal/st.anchor_weights))
                scale-=256*np.finfo(float).eps*(1+abs(scale))
                if same:action='reuse';scale=1.;reason='exact_current_model'
                elif scale>=p.minimum_scale and np.isfinite(scale):action='update';reason='positive_scaled_loewner_bound'
                elif scale>0:
                    reason='valid_update_too_weak';st.metrics['weak_updates']+=1
                else:
                    reason='positive_definiteness_not_proved';st.metrics['invalidations']+=1
            else:
                # A cached unsuccessful envelope is never called a valid spectrum.
                # Skip refinement while change is small; Gray remains the fallback.
                if same or (k-st.anchor_last<p.failed_envelope_age and eps<=.05*max(st.anchor_norm,1e-30)):
                    action='fallback';reason='cached_no_envelope';st.metrics['fallbacks']+=1
                else:reason='failed_envelope_recheck'
        if action in ('reuse','update'):
            cost=8*rx.n
            if not ledger.fits(cost):action='fallback';reason='update_budget';st.metrics['fallbacks']+=1
            else:
                ledger.add('reuse_scaled_envelope',cost)
                self.weights=st.anchor_weights*scale
                self.inverse_directions=st.anchor_directions/scale if st.anchor_directions is not None else None
                st.metrics['reuses' if action=='reuse' else 'updates']+=1
                self.metrics['triggered']=True
        if action=='rebuild':
            cost=100*rx.nnz+5*rx.n*np.log2(rx.n)+80*rx.n
            dirs=20*rx.n*np.log2(rx.n)+40*rx.n
            if ledger.fits(cost+dirs):
                ledger.add('deep_envelope_preparation',cost)
                self.weights,details=circulant_envelope(rx.matrix,rx.noise);self.metrics.update(details)
                if self.weights is not None:
                    ledger.add('deep_inverse_directions',dirs)
                    self.inverse_directions=direction_values(rx.wave,self.weights)
                st.anchor=rx.channel;st.anchor_weights=None if self.weights is None else self.weights.copy()
                st.anchor_directions=None if self.inverse_directions is None else self.inverse_directions.copy()
                st.anchor_norm=channel_norm(rx.channel);st.anchor_noise=rx.noise;st.anchor_last=k;st.metrics['rebuilds']+=1
            else:action='fallback';reason='rebuild_budget';st.metrics['fallbacks']+=1
        self.transition=dict(action=action,reason=reason,delta_c=eps,delta_a_norm_bound=eta,scale=scale,
                             anchor_age=k-st.anchor_last,valid_envelope=self.weights is not None)
        self.metrics.update(self.transition)
        self.metrics['preparation_seconds']+=time.perf_counter()-start
