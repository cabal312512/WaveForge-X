"""R6 current-frame path descriptions and charged reusable sparse assembly.

Only current/past CSI snapshots enter receiver state. The full current operator
is always applied; caching never substitutes an old propagation channel.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import csr_matrix
from .tdl_profile import TDLProfile


def accurate_kernel(delay, half=32):
    """DC-normalized Kaiser-sinc; main occupied band is |omega|<=pi/2.

    The half-band is a real common 2x oversampling resource restriction, not an
    accuracy metric substituted for full-band payload. See OversampledWave.
    """
    base=int(np.floor(delay)); fraction=float(delay-base)
    if delay<0 or not np.isfinite(delay) or half<1:raise ValueError('invalid delay/support')
    j=np.arange(2*half+1)
    h=np.sinc(j-half-fraction)*np.kaiser(len(j),10.)
    if fraction==0:h[:]=0;h[half]=1
    h/=h.sum()
    keep=h!=0
    return base+j[keep],h[keep]


@dataclass(frozen=True)
class PathFrame:
    ids: tuple
    kernels: tuple
    amplitudes: tuple
    frequencies: tuple
    start_index: int
    sample_rate_hz: float
    n: int
    cp: int
    family: str
    doppler_hz: float
    filter_parameters: tuple=()

    def __post_init__(self):
        if self.n<1 or not 0<=self.cp<=self.n or self.start_index<0 or not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz<=0:
            raise ValueError('invalid current frame coordinates')
        if not self.ids or len(set(self.ids))!=len(self.ids) or not len(self.ids)==len(self.kernels)==len(self.amplitudes)==len(self.frequencies):
            raise ValueError('unique matched path descriptors required')
        def own(a,dtype):
            a=np.asarray(a,dtype=dtype)
            if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('finite nonempty path vectors required')
            return np.frombuffer(a.tobytes(),dtype=dtype)
        kernels=tuple((own(d,np.int64),own(h,np.complex128)) for d,h in self.kernels)
        amps=tuple(own(a,np.complex128) for a in self.amplitudes);freq=tuple(own(f,np.float64) for f in self.frequencies)
        if any(len(d)!=len(h) or d.min()<0 or d.max()>self.cp for d,h in kernels) or any(len(a)!=len(f) for a,f in zip(amps,freq)):
            raise ValueError('matched path arrays and sufficient complete guard required')
        if any(len(np.unique(d))!=len(d) for d,h in kernels):
            raise ValueError('canonical path kernels require unique delay offsets')
        object.__setattr__(self,'kernels',kernels);object.__setattr__(self,'amplitudes',amps);object.__setattr__(self,'frequencies',freq)

    @property
    def n_paths(self):return len(self.ids)

    @property
    def max_delay(self):return max(int(d.max()) for d,h in self.kernels)

    def signature(self):
        h=hashlib.sha256()
        for identity,(d,k),f in zip(self.ids,self.kernels,self.frequencies):
            h.update(str(identity).encode());h.update(d.tobytes());h.update(k.tobytes());h.update(f.tobytes())
        h.update(np.array([self.n,self.cp,self.sample_rate_hz]).tobytes())
        return h.hexdigest()

    def coefficient_key(self):
        h=hashlib.sha256(self.signature().encode())
        for a,f in zip(self.amplitudes,self.frequencies):
            h.update((a*np.exp(2j*np.pi*f*self.start_index/self.sample_rate_hz)).tobytes())
        return h.hexdigest()

    def coefficients(self, indices):
        times=np.asarray(indices)/self.sample_rate_hz
        return np.array([a@np.exp(2j*np.pi*f[:,None]*times) for a,f in zip(self.amplitudes,self.frequencies)])

    def apply(self, transmitted):
        x=np.asarray(transmitted,complex);g=self.coefficients(self.start_index+np.arange(len(x)))
        y=np.zeros_like(x)
        for p,(ds,h) in enumerate(self.kernels):
            for d,w in zip(ds,h):
                if d<len(x):y[d:]+=g[p,d:]*w*x[:len(x)-d]
        return y

    def matrix(self,wave,start_index=0):
        from scipy.sparse import coo_matrix
        if self.cp<self.max_delay:raise ValueError('guard shorter than complete support')
        rows=np.arange(self.n);g=self.coefficients(self.start_index+self.cp+rows)
        rr=[];cc=[];vv=[]
        for p,(ds,h) in enumerate(self.kernels):
            for d,w in zip(ds,h):
                source=rows-d;phase=np.ones(self.n,complex);mask=source<0
                phase[mask]=wave.prefix_phases[self.cp+source[mask]]
                rr.append(rows);cc.append(source%self.n);vv.append(g[p]*w*phase)
        c=coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(self.n,self.n)).tocsr()
        c.eliminate_zeros();return c

    def preparation_work(self,n):
        s=sum(len(f) for f in self.frequencies);q=sum(len(d) for d,h in self.kernels);entries=n*q
        return float(40*n*s+entries*(20+2*np.log2(max(2,entries)))+64*q)


def tdl_frame(profile,seed,n,cp,epoch,fd,ds=100e-9,half=32,birth=False):
    c=TDLProfile(profile,seed,ds,fd,7.68e6,half=0 if half==0 else half)
    kernels=tuple(accurate_kernel(d,half) for d in c.delays)
    amplitudes=[];frequencies=[]
    for p in range(c.n_paths):
        if c.los[p]:
            amplitudes.append(np.sqrt(c.power[p])*np.array([np.exp(1j*c.los_phase[p])]))
            frequencies.append(np.array([.7*fd]))
        else:
            amplitudes.append(np.sqrt(c.power[p])*c.amplitudes[p].copy())
            frequencies.append(fd*np.cos(c.angles[p]))
    ids=list(range(c.n_paths))
    # Controlled abrupt event in the same TDL background; observed current CSI.
    if birth:
        for p in [2,4]:
            if epoch>=35:amplitudes[p]*=0.
        if 55<=epoch<80:
            ids.append('birth');kernels+= (accurate_kernel(2.3,half),)
            amplitudes.append(np.array([.45*np.exp(.3j)]));frequencies.append(np.array([fd*.23]))
    parameters=tuple((float(d),half) for d in c.delays)
    if birth and 55<=epoch<80:parameters+=((2.3,half),)
    return PathFrame(tuple(ids),kernels,tuple(amplitudes),tuple(frequencies),epoch*(n+cp),7.68e6,n,cp,'tdl-'+profile,fd,parameters)


def synthetic_frame(seed,n,cp,epoch,family):
    rng=np.random.default_rng(seed)
    ds=np.array([0,2,5,9]);g=(rng.normal(size=4)+1j*rng.normal(size=4))*np.array([1.,.35,.2,.15])
    g/=np.linalg.norm(g)
    fd={'slow':.2,'moderate':40.,'high':1600.,'abrupt':1.}[family]
    frequencies=rng.uniform(-1,1,4)*fd
    ids=list(range(4));kernels=tuple((np.array([d]),np.array([1.])) for d in ds)
    amplitudes=[np.array([a]) for a in g];freq=[np.array([f]) for f in frequencies]
    if family=='abrupt':
        if epoch>=35:amplitudes[1]*=0
        if 55<=epoch<80:
            ids.append('birth');kernels+=((np.array([7]),np.array([1.])),)
            amplitudes.append(np.array([.65*np.exp(.2j)]));freq.append(np.array([.7]))
    return PathFrame(tuple(ids),kernels,tuple(amplitudes),tuple(freq),epoch*(n+cp),64000.,n,cp,family,fd)


def path_change_bound(old,new):
    """Deterministic l1-kernel/phase upper bound of ||C_new-C_old||_2.

    S_d has unit norm for each actual unit-modulus CP/CPP. The same relative
    useful-sample coordinates are compared. Delay, power, frequency and path
    identity changes are accounted for, including missing paths as zero.
    No stored/current dense operator is compared. Exact-arithmetic validity;
    elementary floating guards are not an end-to-end interval proof.
    """
    if (old.n,old.cp,old.sample_rate_hz)!=(new.n,new.cp,new.sample_rate_hz):return np.inf,{}
    oi={v:i for i,v in enumerate(old.ids)};ni={v:i for i,v in enumerate(new.ids)}
    contributions={};values=[];duration=(new.n-1)/new.sample_rate_hz
    for identity in set(oi)|set(ni):
        if identity not in oi or identity not in ni:
            frame,index=(new,ni[identity]) if identity in ni else (old,oi[identity])
            value=float(np.abs(frame.amplitudes[index]).sum()*np.abs(frame.kernels[index][1]).sum())
        else:
            p,q=oi[identity],ni[identity]
            oa,na=old.amplitudes[p],new.amplitudes[q];of,nf=old.frequencies[p],new.frequencies[q]
            old_peak=float(abs(oa).sum());new_peak=float(abs(na).sum())
            od,oh=old.kernels[p];nd,nh=new.kernels[q]
            om={int(d):complex(h) for d,h in zip(od,oh)};nm={int(d):complex(h) for d,h in zip(nd,nh)}
            kernel_delta=sum(abs(om.get(d,0)-nm.get(d,0)) for d in set(om)|set(nm))
            if oa.shape==na.shape:
                # Relative-frame phase anchors plus frequency drift across block.
                ao=oa*np.exp(2j*np.pi*of*(old.start_index+old.cp)/old.sample_rate_hz)
                an=na*np.exp(2j*np.pi*nf*(new.start_index+new.cp)/new.sample_rate_hz)
                change=float(np.sum(abs(an-ao)+abs(ao)*np.minimum(2.,2*np.pi*abs(nf-of)*duration)))
            else:change=old_peak+new_peak
            value=float(change*np.abs(nh).sum()+old_peak*kernel_delta)
        # Accounting must not depend on serialized diagnostic labels: integer
        # 0 and string '0' are distinct legal path identities.
        values.append(value)
        contributions[f'{type(identity).__name__}:{identity}']=value
    result=sum(values)
    return result+128*np.finfo(float).eps*(1+result),contributions


class SparsePathAssembly:
    """CSR indices/kernel mixing and relative SOS phase tables persist.

    N-by-support storage is the sparse operator, never dense G/A. Path kernel
    mixing is P-by-support. Each method owns its cache and pays its first build.
    """
    def __init__(self,wave,frame):
        self.signature=frame.signature();self.n=frame.n
        kernels=tuple(accurate_kernel(d,half) for d,half in frame.filter_parameters) if frame.filter_parameters else frame.kernels
        offsets=np.unique(np.concatenate([d for d,h in kernels]));self.offsets=offsets
        rows=np.arange(frame.n)[:,None];columns=(rows-offsets[None,:])%frame.n
        order=np.argsort(columns,axis=1);self.order=order
        self.indices=np.take_along_axis(columns,order,axis=1).ravel().astype(np.int32)
        self.indptr=(np.arange(frame.n+1)*len(offsets)).astype(np.int32)
        source=rows-offsets[None,:];phase=np.ones(source.shape,complex);mask=source<0
        phase[mask]=wave.prefix_phases[frame.cp+source[mask]];self.phase=phase
        self.mix=np.zeros((frame.n_paths,len(offsets)),complex)
        for p,(d,h) in enumerate(kernels):self.mix[p,np.searchsorted(offsets,d)]=h
        self.tables=tuple(np.exp(2j*np.pi*f[:,None]*np.arange(frame.n)[None,:]/frame.sample_rate_hz) for f in frame.frequencies)
        nnz=frame.n*len(offsets);q=sum(len(d) for d,h in frame.kernels);s=sum(len(f) for f in frame.frequencies)
        self.static_work=float(32*frame.n*s+frame.n*len(offsets)*(20+2*np.log2(max(2,len(offsets))))+12*q+48*frame.cp)

    def build(self,frame):
        gains=np.array([(a*np.exp(2j*np.pi*f*(frame.start_index+frame.cp)/frame.sample_rate_hz))@table for a,f,table in zip(frame.amplitudes,frame.frequencies,self.tables)])
        values=(gains.T@self.mix)*self.phase
        data=np.take_along_axis(values,self.order,axis=1).ravel()
        c=csr_matrix((data,self.indices,self.indptr),shape=(frame.n,frame.n),copy=True)
        s=sum(len(f) for f in frame.frequencies);d=len(self.offsets)
        cost=8*frame.n*s+40*s+8*frame.n*frame.n_paths*d+6*frame.n*d
        return c,float(cost)


class OversampledWave:
    """Unitary completion of a 2x oversampled existing waveform.

    First M=N/2 coordinates carry M QAM payload symbols; remaining coordinates
    are zero and consume real communication resources. Only payload bits are
    delivered. Full-N LMMSE is the specified linear reference, not the reduced-
    prior optimum exploiting the nulls. Certifying all N coordinates with delta
    scaled by M/N conservatively guarantees the returned M-coordinate bits.
    """
    def __init__(self,base,n,cp):
        self.base=base;self.name=base.name;self.n_symbols=n;self.cp_length=cp;self.payload_symbols=base.n_symbols
        m=base.n_symbols
        self.occupied=np.mod(np.fft.fftfreq(m)*m,n).astype(int)
        self.unused=np.setdiff1d(np.arange(n),self.occupied)
        self.subcarriers=getattr(base,'subcarriers',m);self.time_slots=getattr(base,'time_slots',1)
    @property
    def prefix_phases(self):
        # Main experiment uses the existing commensurate even-length AFDM
        # default, whose coarse CPP is a CP. Extend that physical periodic pulse
        # with a common oversampled CP; do not infer arbitrary CPP equivalence.
        if self.name=='afdm' and not (float(2*self.base.c1*self.payload_symbols).is_integer()
                                     and float(self.base.c1*self.payload_symbols**2).is_integer()):
            raise ValueError('oversampled study supports only commensurate AFDM CPP')
        return np.ones(self.cp_length,complex)
    def synthesis(self,s):
        m=self.payload_symbols;f=np.empty(self.n_symbols,complex)
        f[self.occupied]=np.fft.fft(self.base.synthesis(np.asarray(s)[:m]),norm='ortho')
        f[self.unused]=np.asarray(s)[m:]
        return np.fft.ifft(f,norm='ortho')
    def analysis(self,x):
        f=np.fft.fft(x,norm='ortho')
        return np.r_[self.base.analysis(np.fft.ifft(f[self.occupied],norm='ortho')),f[self.unused]]
    def modulate(self,s):
        x=self.synthesis(s)
        return np.r_[self.prefix_phases*x[-self.cp_length:],x]
