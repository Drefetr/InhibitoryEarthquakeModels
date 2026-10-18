"""Normalized temporal kernels for an independent empirical reconstruction.

The ML implementation is explicitly a positive, normalized finite exponential
approximation. It is not the original manuscript's numerical implementation.
Its finite value h(0) does not reproduce the true ML singularity at zero.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from scipy.special import erfcx


@dataclass(frozen=True)
class KernelSpec:
    kind: str
    tau: float
    alpha: float = .5
    p: float = 1.3
    spectral_step: float = .4
    spectral_tail_bound: float = 1e-8

    def __post_init__(self):
        if self.kind not in ('exp', 'ml', 'omori') or self.tau <= 0:
            raise ValueError('unknown kernel or nonpositive scale')
        if self.kind == 'ml' and not 0 < self.alpha < 1:
            raise ValueError('ML alpha must be in (0,1)')
        if self.kind == 'omori' and self.p <= 1:
            raise ValueError('normalized Omori requires p>1')

    @property
    def key(self):
        return (self.kind, self.tau, self.alpha if self.kind == 'ml' else self.p if self.kind == 'omori' else 0.,
                self.spectral_step if self.kind=='ml' else None,
                self.spectral_tail_bound if self.kind=='ml' else None)

    @cached_property
    def mixture(self):
        if self.kind == 'exp':
            return np.array([1.]), np.array([1./self.tau])
        if self.kind != 'ml':
            raise ValueError('Omori is evaluated directly')
        a = self.alpha
        coefficient = 4*np.sin(np.pi*a)/(np.pi*a)
        extent = max(np.log(4.)/a, np.log(coefficient/self.spectral_tail_bound)/a)
        count = int(np.ceil(extent/self.spectral_step))
        v = np.arange(-count, count+1)*self.spectral_step
        raw = self.spectral_step*np.sin(np.pi*a)/(2*np.pi*(np.cosh(a*v)+np.cos(np.pi*a)))
        raw[[0,-1]] *= .5
        return raw/raw.sum(), np.exp(v)/self.tau

    @property
    def approximation_metadata(self):
        if self.kind != 'ml':
            return {'kind': self.kind, 'normalized': True, 'finite_mixture_approximation': False}
        a=self.alpha
        coefficient=4*np.sin(np.pi*a)/(np.pi*a)
        count=int(np.ceil(max(np.log(4.)/a,np.log(coefficient/self.spectral_tail_bound)/a)/self.spectral_step))
        v=np.arange(-count,count+1)*self.spectral_step
        raw=self.spectral_step*np.sin(np.pi*a)/(2*np.pi*(np.cosh(a*v)+np.cos(np.pi*a)))
        raw[[0,-1]]*=.5
        return {'kind':'ml', 'finite_mixture_approximation':True, 'modes':len(v),
                'log_rate_extent':float(v[-1]),'spectral_step':self.spectral_step,
                'raw_mass':float(raw.sum()),'omitted_spectral_mass_bound':float(coefficient*np.exp(-a*v[-1])),
                'normalized':True,'origin_singularity_reproduced':False}

    def _mixture_value(self, lag, derivative=False, integral=False):
        x=np.asarray(lag,dtype=float)
        if np.any(x<0):
            raise ValueError('negative lag')
        weights,rates=self.mixture
        flat=x.ravel()
        out=np.empty_like(flat)
        for start in range(0,len(flat),4096):
            q=flat[start:start+4096,None]*rates
            if integral:
                out[start:start+4096]=(-np.expm1(-q))@(weights/rates)
            else:
                out[start:start+4096]=np.exp(-q)@(weights*rates if derivative else weights)
        out=out.reshape(x.shape)
        return float(out) if x.ndim==0 else out

    def phi(self, lag):
        if self.kind=='omori':
            x=np.asarray(lag,dtype=float)
            if np.any(x<0): raise ValueError('negative lag')
            return np.exp((1-self.p)*np.log1p(x/self.tau))
        return self._mixture_value(lag)

    def h(self, lag):
        if self.kind=='omori':
            x=np.asarray(lag,dtype=float)
            if np.any(x<0): raise ValueError('negative lag')
            return (self.p-1)/self.tau*np.exp(-self.p*np.log1p(x/self.tau))
        return self._mixture_value(lag,derivative=True)

    def cdf(self, lag):
        if self.kind=='omori':
            return -np.expm1((1-self.p)*np.log1p(np.asarray(lag)/self.tau))
        x=np.asarray(lag,dtype=float)
        weights,rates=self.mixture
        flat=x.ravel();out=np.empty_like(flat)
        for start in range(0,len(flat),4096):
            out[start:start+4096]=(-np.expm1(-flat[start:start+4096,None]*rates))@weights
        out=out.reshape(x.shape)
        return float(out) if x.ndim==0 else out

    def A(self, lag):
        """Integrated relaxation, for the distinct physical trend closure."""
        if self.kind=='omori':
            x=np.asarray(lag)/self.tau
            if abs(self.p-2)<1e-12: return self.tau*np.log1p(x)
            return self.tau*np.expm1((2-self.p)*np.log1p(x))/(2-self.p)
        return self._mixture_value(lag,integral=True)


def ml_halforder_phi(lag, tau=1.):
    return erfcx(np.sqrt(np.asarray(lag)/tau))


def ml_halforder_h(lag, tau=1.):
    """Independent derivative formula, with cancellation-safe large-lag series."""
    x=np.sqrt(np.asarray(lag,dtype=float)/tau)
    with np.errstate(divide='ignore'):
        regular=(1/(np.sqrt(np.pi)*x)-erfcx(x))/tau
        large=(1-1.5/x**2+3.75/x**4-13.125/x**6+59.0625/x**8)/(2*np.sqrt(np.pi)*tau*x**3)
    return np.where(x>30,large,regular)
