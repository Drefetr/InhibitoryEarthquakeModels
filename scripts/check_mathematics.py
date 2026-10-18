#!/usr/bin/env python3
"""Deterministic checks of the v3 referee objections.

No catalogue fitting, process simulation, bootstrap, manuscript edit, or
historical result regeneration is performed. Outputs are written beside this
script, unless --output is supplied. Dependencies: mpmath, sympy, numpy, scipy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad
import sympy as sp


def exact_transform_check() -> dict:
    # Scalar symbols represent Laplace transforms of causal kernels/measures.
    # The same computation is valid in the locally integrable causal convolution
    # algebra; no stationary or infinite-history assertion is being made.
    ph, h, gamma, n, m1, x0, rho, s, ng, nk, psi = sp.symbols(
        'ph h gamma n m1 x0 rho s ng nk psi')
    R = ph / (1 - n*h + gamma*ph)
    Q = R * (1 - n*h)
    X = (x0 + rho/s)*ph - ph*ng
    lam = psi + h*nk
    mg = ng - m1*lam
    mk = nk - n*lam
    rhs = (x0+rho/s)*Q + R*(gamma*X-m1*psi) - Q*mg - m1*R*h*mk
    residual = sp.factor(sp.cancel(rhs-X))
    relation_1 = sp.factor(sp.cancel(Q + gamma*R*ph - ph))
    relation_2 = sp.factor(sp.cancel(Q + n*R*h - R))
    assert residual == 0 and relation_1 == 0 and relation_2 == 0
    wrong_rhs = (x0+rho/s)*ph + R*(gamma*X-m1*psi) - Q*mg - m1*R*h*mk
    wrong_residual = sp.factor(sp.cancel(wrong_rhs-X))
    assert sp.cancel(wrong_residual-(x0+rho/s)*gamma*R*ph) == 0
    return {
        'equation_3_13_residual': str(residual),
        'Q_plus_gamma_R_Phi_minus_Phi': str(relation_1),
        'Q_plus_n_R_h_minus_R': str(relation_2),
        'reviewer_Phi_substitution_error': str(wrong_residual),
        'interpretation': 'Exact scalar convolution algebra, not a stochastic existence proof.'
    }


def fractional_no_event_check() -> dict:
    mp.mp.dps = 55
    root1 = (-1 + mp.sqrt(3)*1j)/2
    root2 = mp.conj(root1)
    def R(t):
        # Inverse Laplace of 1/(s + sqrt(s) + 1), via partial fractions in sqrt(s).
        if t == 0:
            return mp.mpf(1)
        def E(root):
            return mp.exp(root*root*t) * mp.erfc(-root*mp.sqrt(t))
        return mp.re((root1*E(root1)-root2*E(root2))/(root1-root2))
    def phi(t):
        return mp.exp(t)*mp.erfc(mp.sqrt(t))
    def iphi(t):
        return phi(t) - 1 + 2*mp.sqrt(t/mp.pi)
    def X(t):
        return phi(t) + iphi(t)
    out=[]
    for t in map(mp.mpf, ['0.1','0.5','1']):
        rq = R(t)
        ir = mp.quad(R, [0,t])
        # No jumps is not a zero compensated martingale:
        # dM_g=-lambda dt=-X dt for n=0, m1=1, Psi(X)=X on this path.
        compensator_term = mp.quad(lambda u: R(t-u)*X(u), [0,t])
        actual=X(t)
        full_rhs=rq+ir+compensator_term
        inv_R=mp.invertlaplace(lambda z: 1/(z+mp.sqrt(z)+1),t,method='dehoog')
        assert abs(inv_R-rq)<mp.mpf('1e-42')
        assert abs(full_rhs-actual)<mp.mpf('1e-42')
        out.append({
            't':float(t), 'Phi':mp.nstr(phi(t),40),
            'integral_Phi':mp.nstr(iphi(t),40), 'Q_equals_R':mp.nstr(rq,40),
            'integral_Q':mp.nstr(ir,40), 'actual_X':mp.nstr(actual,40),
            'rhs_with_martingale_term_omitted':mp.nstr(rq+ir,40),
            'omitted_compensator_contribution':mp.nstr(compensator_term,40),
            'full_rhs':mp.nstr(full_rhs,40), 'absolute_residual':mp.nstr(abs(full_rhs-actual),10),
            'wrong_Phi_replacement_full_rhs':mp.nstr(actual+compensator_term,40),
            'R_closed_form_vs_inverse_laplace_abs_error':mp.nstr(abs(inv_R-rq),10)
        })
    return {'parameters': {'n':0,'m1':1,'ell':1,'gamma':1,'alpha':0.5,'tau':1,'X0':1,'rho':1},
            'checks':out,
            'scope':'Chosen deterministic no-event path. The linear link is only needed along its positive stress range; it may be extended positively outside that range.'}


def product_derivative_checks() -> dict:
    # Chosen deterministic event histories, not simulated catalogues.
    times=np.array([0.15,0.40,0.75])
    marks_g=np.array([50.0,4.0,0.2])
    marks_k=np.array([2.0,1.0,3.0])
    T=1.0
    Bbar=1.125
    def B(t): return 1.0+0.25*t
    def history(t):
        idx=times<t
        lags=t-times[idx]
        return (float(np.dot(marks_g[idx],np.exp(-1.3*lags))),
                float(np.dot(marks_k[idx],0.9*np.exp(-0.9*lags))))
    G=np.array([history(t)[0] for t in times])
    H=np.array([history(t)[1] for t in times])
    Bi=1+0.25*times
    cuts=[0.0,*times,T]
    def integ(f):
        return sum(quad(f,l,r,epsabs=1e-12,epsrel=1e-12)[0] for l,r in zip(cuts[:-1],cuts[1:]))
    def calc(eta,n):
        J=integ(lambda t: B(t)*(-math.expm1(-eta*history(t)[0])))
        Jp=integ(lambda t: B(t)*history(t)[0]*math.exp(-eta*history(t)[0]))
        ref=Bi+n*H
        ratios=(Bi*np.exp(-eta*G)+n*H)/ref
        ratios_p=-Bi*G*np.exp(-eta*G)/ref
        product=float(np.prod(ratios))
        deriv=math.exp(J)*(Jp*product+sum(float(ratios_p[i]*np.prod(np.delete(ratios,i))) for i in range(len(times))))
        return math.exp(J)*product, deriv
    out=[]
    for n in [0.0,0.4]:
        for eta in [0.3,1.0,2.0]:
            eta_min=eta/2
            value,deriv=calc(eta,n)
            dx=1e-5
            fd=(calc(eta+dx,n)[0]-calc(eta-dx,n)[0])/(2*dx)
            envelope=math.exp(Bbar)*(Bbar+len(times))/(math.e*eta_min)
            assert abs(deriv)<=envelope
            assert math.isclose(deriv,fd,rel_tol=2e-6,abs_tol=2e-9)
            out.append({'n':n,'eta':eta,'L':value,'direct_product_derivative':deriv,
                        'central_finite_difference':fd,'claimed_uniform_envelope':envelope,
                        'envelope_pass':True})
    return {'cases':out, 'single_factor_example': {
        'B':1,'triggering':0,'G':100,'eta':1,
        'absolute_log_factor_derivative':100,
        'absolute_factor_derivative':100*math.exp(-100),
        'one_over_e_eta':1/math.e,
        'explanation':'The large log derivative is not the derivative of the likelihood factor.'}}


def paired_mcse_checks() -> list:
    out=[]
    for positive,negative in [(29,21),(61,6)]:
        N=150
        mean=(positive-negative)/N
        q=positive+negative
        sum_centered=q-N*mean*mean
        sample_se=math.sqrt(sum_centered/(N*(N-1)))
        plugin_se=math.sqrt(sum_centered/(N*N))
        mcnemar_null_se=math.sqrt(q)/N
        out.append({'N':N,'positive_discordances':positive,'negative_discordances':negative,
                    'mean_difference_pp':100*mean,
                    'sample_variance_mcse_pp':100*sample_se,
                    'plugin_mcse_pp':100*plugin_se,
                    'mcnemar_null_se_pp':100*mcnemar_null_se})
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path(__file__).with_name('check_results.json'))
    args=ap.parse_args()
    beta=2.849401764894274
    r=0.7
    gamma=0.75*math.log(10)
    result={
        'scope':'Deterministic algebra, quadrature, and arithmetic. No manuscript edits or scientific campaign runs.',
        'versions':{'python':platform.python_version(),'mpmath':mp.__version__,'sympy':sp.__version__,'numpy':np.__version__,'scipy':scipy.__version__},
        'exact_transform_algebra':exact_transform_check(),
        'fractional_no_event':fractional_no_event_check(),
        'likelihood_product_derivative':product_derivative_checks(),
        'paired_mcse':paired_mcse_checks(),
        'reviewer_alternative_envelope_moment':{
            'beta':beta,'r':r,'gamma':gamma,'gamma_plus_r_beta':gamma+r*beta,
            'Eg_finite':gamma<beta,'Ekappa_finite':r<1,
            'Eg_kappa_finite':gamma+r*beta<beta,
            'reason':'E[g*kappa] diverges under the declared synthetic exponential-mark law. Under the noninhibited Hawkes law, immigrant-to-direct-child contributions can therefore make E sum_i G(T_i-) infinite despite finite mean counts and first mark moments.'
        }
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
