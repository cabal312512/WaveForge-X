"""Cache constant Gray labels/cuts; exact same R4 shared-energy formulas.

No center, mask, residual, costs or DP result survives a frame. Tests compare
the specialized functions with the original functions, including boundary ties.
"""
import numpy as np
from .certiphy import CertiPHY
from types import FunctionType


class GrayGeometry:
    def __init__(self,width):
        self.width=width;self.p=max(1,width//2);self.side=2**self.p
        self.scale=float(np.sqrt(2*(2**width-1)/3))
        self.cuts=(2*np.arange(1,self.side)-self.side)/self.scale
        labels=np.arange(self.side)^(np.arange(self.side)>>1)
        self.labels=labels;self.boundaries=np.r_[-np.inf,self.cuts,np.inf]
        toggles=labels[:-1]^labels[1:]
        self.relevant=tuple(self.cuts[(toggles&(1<<(self.p-1-bit)))!=0] for bit in range(self.p))
        self.shifts=1<<np.arange(self.p-1,-1,-1)
        self.popcount=np.array([int(i).bit_count() for i in range(self.side)])
        self.counts=np.arange(self.p+1)
        # Private globals specialization preserves the frozen solve bytecode;
        # it does not monkey-patch process-global functions or another receiver.
        globals_map=dict(CertiPHY.solve.__globals__)
        globals_map.update(bit_margins=self.margins,targeted_gray_bound=self.targeted)
        self.solve=FunctionType(CertiPHY.solve.__code__,globals_map,'cached_certiphy_solve',CertiPHY.solve.__defaults__,CertiPHY.solve.__closure__)

    def margins(self,symbols,modulation):
        values=np.asarray(symbols)
        if self.width==1:return abs(values.real)[:,None]
        result=np.empty((len(values),self.width))
        for offset,x in [(0,values.real),(self.p,values.imag)]:
            for bit,cuts in enumerate(self.relevant):result[:,offset+bit]=abs(x[:,None]-cuts).min(axis=1)
        return result

    def gray(self,symbols,radius,active,target,margins):
        remaining=int(active.sum());energy=float(radius)**2
        if not remaining:return 0
        if self.width<=2:
            return int(np.searchsorted(np.cumsum(np.sort(margins[active]**2)),energy,side='right'))
        coordinates=np.r_[symbols.real,symbols.imag];masks=np.r_[active[:,:self.p],active[:,self.p:]]
        mask_labels=masks@self.shifts
        natural=np.clip(np.floor((coordinates*self.scale+self.side)/2),0,self.side-1).astype(int)
        current=natural^(natural>>1);costs=np.full((len(coordinates),self.p+1),np.inf);costs[:,0]=0.
        rows=np.arange(len(coordinates))
        for q,label in enumerate(self.labels):
            count=self.popcount[(current^label)&mask_labels]
            distance=np.maximum(np.maximum(self.boundaries[q]-coordinates,coordinates-self.boundaries[q+1]),0)
            np.minimum.at(costs,(rows,count),distance**2)
        available=margins[active]**2;anchor=np.partition(available,min(target,len(available)-1))[min(target,len(available)-1)]
        if anchor<=np.finfo(float).tiny:return remaining
        best=float(remaining)
        for factor in [.25,.5,1.,2.,4.]:
            multiplier=factor/max(anchor,1e-300)
            best=min(best,float(multiplier*energy+np.max(self.counts[None,:]-multiplier*costs,axis=1).sum()))
        return min(remaining,max(0,int(np.floor(best+1e-9))))

    def targeted(self,symbols,modulation,radius,active,target,margins):
        remaining=int(active.sum())
        if remaining<=target:return remaining,False
        if self.width<=2:
            smallest=np.partition(margins[active]**2,target)[:target+1]
            return (target if smallest.sum()>radius**2 else remaining),False
        axis=np.minimum(np.where(active,margins,np.inf).reshape(len(symbols),2,self.p).min(axis=2),np.inf).ravel()**2
        if len(axis)>target and np.partition(axis,target)[:target+1].sum()<=radius**2:return remaining,False
        return self.gray(symbols,radius,active,target,margins),True
