"""cr2_methods.py

Small-sample cluster-robust inference for GLMs (CR2 + Satterthwaite), following
Bell & McCaffrey (2002) and Pustejovsky & Tipton (2018), as implemented in R clubSandwich
for lm/glm (working-weight linearisation). Also: Mancl-DeRouen, CR1, ordinal logit,
Brant test, random-intercept models (Gauss-Hermite), Firth logistic with profile CIs."""
import numpy as np, pandas as pd
from scipy import stats, optimize, linalg

# ---------------- GLM via IRLS ----------------
def fit_glm(y, X, fam, pw=None, tol=1e-10):
    n, k = X.shape
    pw = np.ones(n) if pw is None else pw
    b = np.zeros(k)
    if fam == 'poisson': b[0] = np.log(np.average(y, weights=pw))
    for _ in range(200):
        eta = X @ b
        if fam == 'poisson': mu = np.exp(eta); dmu = mu; v = mu
        elif fam == 'logit': mu = 1/(1+np.exp(-eta)); dmu = mu*(1-mu); v = dmu
        else: mu = eta; dmu = np.ones(n); v = np.ones(n)
        w = pw*dmu**2/v
        z = eta + (y-mu)/dmu
        bn = np.linalg.solve(X.T @ (X*w[:, None]), X.T @ (w*z))
        if np.max(np.abs(bn-b)) < tol: b = bn; break
        b = bn
    eta = X @ b
    if fam == 'poisson': mu = np.exp(eta); dmu = mu; v = mu
    elif fam == 'logit': mu = 1/(1+np.exp(-eta)); dmu = mu*(1-mu); v = dmu
    else: mu = eta; dmu = np.ones(n); v = np.ones(n)
    w = pw*dmu**2/v
    rt = np.sqrt(w)*(y-mu)/dmu          # residual on the linearised (whitened) scale
    Xt = X*np.sqrt(w)[:, None]
    return b, Xt, rt

def _isqrt_psd(M):
    """symmetric (pseudo-)inverse square root"""
    vals, vecs = np.linalg.eigh((M+M.T)/2)
    inv = np.array([1/np.sqrt(v) if v > 1e-10 else 0.0 for v in vals])
    return (vecs*inv) @ vecs.T

def cr2(b, Xt, rt, cl, c):
    """returns estimate, se, Satterthwaite df for contrast c'b"""
    B = np.linalg.inv(Xt.T @ Xt)
    n = len(rt); groups = pd.unique(cl)
    H = Xt @ B @ Xt.T
    IH = np.eye(n) - H
    meat = np.zeros_like(B); Q = []
    Bc = B @ c
    for g in groups:
        i = np.where(cl == g)[0]
        A = _isqrt_psd(IH[np.ix_(i, i)])
        u = Xt[i].T @ (A @ rt[i]); meat += np.outer(u, u)
        gg = A @ (Xt[i] @ Bc)
        Q.append(IH[i, :].T @ gg)
    V = B @ meat @ B
    Q = np.column_stack(Q); Om = Q.T @ Q
    df = np.trace(Om)**2/np.trace(Om @ Om)
    return c @ b, np.sqrt(c @ V @ c), df

def cr_md(b, Xt, rt, cl, c):
    """Mancl-DeRouen (CR3-type) with t(G-1)"""
    B = np.linalg.inv(Xt.T @ Xt); meat = np.zeros_like(B); groups = pd.unique(cl)
    for g in groups:
        i = np.where(cl == g)[0]; Hgg = Xt[i] @ B @ Xt[i].T
        u = Xt[i].T @ np.linalg.pinv(np.eye(len(i))-Hgg) @ rt[i]; meat += np.outer(u, u)
    V = B @ meat @ B
    return c @ b, np.sqrt(c @ V @ c), len(groups)-1

def cr1(b, Xt, rt, cl, c):
    B = np.linalg.inv(Xt.T @ Xt); meat = np.zeros_like(B); groups = pd.unique(cl)
    for g in groups:
        i = np.where(cl == g)[0]; u = Xt[i].T @ rt[i]; meat += np.outer(u, u)
    G = len(groups); n, k = Xt.shape
    V = B @ meat @ B*G/(G-1)*(n-1)/(n-k)
    return c @ b, np.sqrt(c @ V @ c), G-1

def hc0(b, Xt, rt, cl, c):
    B = np.linalg.inv(Xt.T @ Xt); U = Xt*rt[:, None]; V = B @ (U.T @ U) @ B
    return c @ b, np.sqrt(c @ V @ c), np.inf

def ci(est, se, df, expo=True):
    t = stats.norm.ppf(.975) if np.isinf(df) else stats.t.ppf(.975, df)
    p = 2*(stats.norm.sf(abs(est/se)) if np.isinf(df) else stats.t.sf(abs(est/se), df))
    f = np.exp if expo else (lambda z: z)
    return dict(est=f(est), lo=f(est-t*se), hi=f(est+t*se), p=p, df=df)

def glm_contrast(y, X, cl, fam, c, method='cr2', pw=None, expo=True):
    b, Xt, rt = fit_glm(y, X, fam, pw)
    fn = dict(cr2=cr2, md=cr_md, cr1=cr1, hc0=hc0)[method]
    return ci(*fn(b, Xt, rt, cl, c), expo=expo)

# ---------------- ordinal (proportional odds) ----------------
def _ord_ll(theta, X, y, K):
    a = np.cumsum(np.r_[theta[0], np.exp(theta[1:K-1])])  # increasing cutpoints
    beta = theta[K-1:]; xb = X @ beta
    F = lambda t: 1/(1+np.exp(-t))
    cum = np.column_stack([F(a[j]-xb) for j in range(K-1)])
    P = np.column_stack([cum[:, 0]] + [cum[:, j]-cum[:, j-1] for j in range(1, K-1)] + [1-cum[:, -1]])
    return np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))

def fit_ordinal(X, y, cl=None):
    K = int(y.max())+1
    th0 = np.r_[-0.5, np.zeros(K-2), np.zeros(X.shape[1])]
    f = lambda t: -_ord_ll(t, X, y, K).sum()
    r = optimize.minimize(f, th0, method='BFGS', options=dict(gtol=1e-8, maxiter=5000))
    th = r.x
    # numerical scores and Hessian
    h = 1e-5; p = len(th)
    S = np.array([(_ord_ll(th+h*e, X, y, K)-_ord_ll(th-h*e, X, y, K))/(2*h) for e in np.eye(p)]).T
    Hs = np.zeros((p, p))
    for i in range(p):
        ei = np.eye(p)[i]
        Hs[i] = (np.array([(_ord_ll(th+h*ei+h*e, X, y, K)-_ord_ll(th+h*ei-h*e, X, y, K)).sum()/(2*h) for e in np.eye(p)]) -
                 np.array([(_ord_ll(th-h*ei+h*e, X, y, K)-_ord_ll(th-h*ei-h*e, X, y, K)).sum()/(2*h) for e in np.eye(p)]))/(2*h)
    Ai = np.linalg.inv(-(Hs+Hs.T)/2)
    return th, Ai, S, K

def ordinal_contrast(X, y, cl, j, method='cr1'):
    th, Ai, S, K = fit_ordinal(X, y)
    idx = K-1+j
    groups = pd.unique(cl); G = len(groups); n, p = S.shape
    if method == 'cr1':
        M = sum(np.outer(S[cl == g].sum(0), S[cl == g].sum(0)) for g in groups)
        V = Ai @ M @ Ai*G/(G-1); df = G-1
    elif method == 'hc0':
        V = Ai @ (S.T @ S) @ Ai; df = np.inf
    elif method == 'model':
        V = Ai; df = np.inf
    # positive beta = higher severity  (P(Y<=j)=F(a_j - xb))
    return ci(th[idx], np.sqrt(V[idx, idx]), df)

def brant(X, y):
    """Brant (1990) test. X without intercept. returns omnibus chi2,df,p and per-variable p."""
    K = int(y.max())+1; n, k = X.shape
    Xc = np.column_stack([np.ones(n), X])
    bs, pis = [], []
    for j in range(K-1):
        yy = (y > j).astype(float)
        b, _, _ = fit_glm(yy, Xc, 'logit'); bs.append(b); pis.append(1/(1+np.exp(-Xc @ b)))
    Vs = {}
    for a in range(K-1):
        for c_ in range(a, K-1):
            Wa = pis[a]*(1-pis[a]); Wc = pis[c_]*(1-pis[c_])
            Ia = np.linalg.inv(Xc.T @ (Xc*Wa[:, None])); Ic = np.linalg.inv(Xc.T @ (Xc*Wc[:, None]))
            Wac = pis[c_]-pis[a]*pis[c_]   # pi_c <= pi_a for c>a  (P(Y>c) <= P(Y>a))
            Vs[(a, c_)] = Ia @ Xc.T @ (Xc*Wac[:, None]) @ Ic
    # stack slopes (drop intercepts)
    beta = np.concatenate([b[1:] for b in bs])
    J = K-1; Vfull = np.zeros((J*k, J*k))
    for a in range(J):
        for c_ in range(J):
            m = Vs[(min(a, c_), max(a, c_))]
            if a > c_: m = m.T
            Vfull[a*k:(a+1)*k, c_*k:(c_+1)*k] = m[1:, 1:]
    D = np.zeros(((J-1)*k, J*k))
    for r in range(J-1):
        D[r*k:(r+1)*k, 0:k] = np.eye(k); D[r*k:(r+1)*k, (r+1)*k:(r+2)*k] = -np.eye(k)
    d = D @ beta; Vd = D @ Vfull @ D.T
    chi = d @ np.linalg.solve(Vd, d); dfo = (J-1)*k
    per = []
    for v in range(k):
        rows = [r*k+v for r in range(J-1)]
        dv = d[rows]; Vv = Vd[np.ix_(rows, rows)]
        cv = dv @ np.linalg.solve(Vv, dv); per.append(stats.chi2.sf(cv, J-1))
    return chi, dfo, stats.chi2.sf(chi, dfo), per, [b[1:] for b in bs]

# ---------------- random intercept models (Gauss-Hermite) ----------------
GH_X, GH_W = np.polynomial.hermite.hermgauss(60)
def ri_fit(ll_fun, theta0, cl):
    """ll_fun(theta_fixed, offset_vector) -> per-observation loglik; last element of theta = log sigma"""
    groups = pd.unique(cl); idx = [np.where(cl == g)[0] for g in groups]
    def negll(th):
        s = np.exp(th[-1]); tot = 0
        for i in idx:
            vals = np.array([ll_fun(th[:-1], i, np.sqrt(2)*s*x).sum() for x in GH_X])
            m = vals.max(); tot += m+np.log(np.sum(GH_W*np.exp(vals-m))/np.sqrt(np.pi))
        return -tot
    r = optimize.minimize(negll, theta0, method='BFGS', options=dict(gtol=1e-6, maxiter=3000))
    th = r.x; p = len(th); h = 1e-4
    Hs = np.zeros((p, p))
    for a in range(p):
        for c_ in range(a, p):
            ea = np.eye(p)[a]*h; ec = np.eye(p)[c_]*h
            Hs[a, c_] = Hs[c_, a] = (negll(th+ea+ec)-negll(th+ea-ec)-negll(th-ea+ec)+negll(th-ea-ec))/(4*h*h)
    return th, np.linalg.inv(Hs)

def ri_logit(y, X, cl):
    def ll(beta, i, off):
        eta = X[i] @ beta+off; return y[i]*eta-np.log1p(np.exp(eta))
    b0, _, _ = fit_glm(y, X, 'logit')
    return ri_fit(ll, np.r_[b0, np.log(0.2)], cl)

def ri_ordinal(y, X, cl):
    K = int(y.max())+1
    def ll(theta, i, off):
        return _ord_ll_off(theta, X[i], y[i], K, off)
    th0, _, _, _ = fit_ordinal(X, y)
    return ri_fit(ll, np.r_[th0, np.log(0.2)], cl), K

def _ord_ll_off(theta, X, y, K, off):
    a = np.cumsum(np.r_[theta[0], np.exp(theta[1:K-1])]); beta = theta[K-1:]; xb = X @ beta+off
    F = lambda t: 1/(1+np.exp(-t))
    cum = np.column_stack([F(a[j]-xb) for j in range(K-1)])
    P = np.column_stack([cum[:, 0]] + [cum[:, j]-cum[:, j-1] for j in range(1, K-1)] + [1-cum[:, -1]])
    return np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))

# ---------------- Firth logistic with profile penalised likelihood CI ----------------
def _firth_pll(b, X, y):
    eta = X @ b; mu = 1/(1+np.exp(-eta)); w = mu*(1-mu)
    ll = np.sum(y*eta-np.log1p(np.exp(eta)))
    sign, logdet = np.linalg.slogdet(X.T @ (X*w[:, None]))
    return ll+0.5*logdet

def firth(y, X, fix=None):
    k = X.shape[1]; b = np.zeros(k)
    free = [i for i in range(k) if fix is None or i != fix[0]]
    if fix is not None: b[fix[0]] = fix[1]
    for _ in range(300):
        eta = X @ b; mu = 1/(1+np.exp(-eta)); w = mu*(1-mu)
        XW = X*np.sqrt(w)[:, None]; I = X.T @ (X*w[:, None])
        h = np.einsum('ij,jk,ik->i', XW, np.linalg.inv(I), XW)
        U = X.T @ (y-mu+h*(0.5-mu))
        Uf = U[free]; If = I[np.ix_(free, free)]
        step = np.linalg.solve(If, Uf)
        # step halving to ensure increase
        t = 1.0; old = _firth_pll(b, X, y)
        while t > 1e-6:
            bn = b.copy(); bn[free] = b[free]+t*step
            if _firth_pll(bn, X, y) >= old-1e-12: break
            t /= 2
        b = bn
        if np.max(np.abs(t*step)) < 1e-9: break
    return b, _firth_pll(b, X, y)

def firth_ci(y, X, j):
    b, pl = firth(y, X)
    crit = stats.chi2.ppf(.95, 1)/2
    g = lambda v: pl-firth(y, X, fix=(j, v))[1]-crit
    def bracket(sign):
        step = 0.1
        while g(b[j]+sign*step) < 0: step *= 1.5
        return b[j]+sign*step
    lo = optimize.brentq(g, bracket(-1), b[j]); hi = optimize.brentq(g, b[j], bracket(1))
    p = stats.chi2.sf(2*(pl-firth(y, X, fix=(j, 0.0))[1]), 1)
    return dict(est=np.exp(b[j]), lo=np.exp(lo), hi=np.exp(hi), p=p)

# ---------------- splines ----------------
def rcs_basis(x, knots):
    t = np.asarray(knots); k = len(t); f = lambda u: np.clip(u, 0, None)**3
    cols = []
    for j in range(k-2):
        cols.append((f(x-t[j]) - f(x-t[k-2])*(t[k-1]-t[j])/(t[k-1]-t[k-2]) + f(x-t[k-1])*(t[k-2]-t[j])/(t[k-1]-t[k-2]))/(t[k-1]-t[0])**2)
    return np.column_stack(cols)

def icc_anova(y, g):
    df = pd.DataFrame({'y': y, 'g': g}); k = df.groupby('g').size(); N = len(df); a = len(k)
    n0 = (N-(k**2).sum()/N)/(a-1)
    msb = (k*(df.groupby('g').y.mean()-df.y.mean())**2).sum()/(a-1)
    msw = ((df.y-df.groupby('g').y.transform('mean'))**2).sum()/(N-a)
    return (msb-msw)/(msb+(n0-1)*msw)

# ---------------- partial proportional odds (one covariate non-proportional), 3 levels ----------------
def _ppo_ll(th, X, z, y):
    a1, a2 = th[0], th[0]+np.exp(th[1]); beta = th[2:2+X.shape[1]]; g1, g2 = th[-2], th[-1]
    F = lambda t: 1/(1+np.exp(-t)); xb = X @ beta
    c1 = F(a1-xb-g1*z); c2 = F(a2-xb-g2*z)
    P = np.column_stack([c1, c2-c1, 1-c2])
    return np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))

def ppo_contrast(X, z, y, cl, j=0):
    p = 2+X.shape[1]+2; th0 = np.r_[-0.5, 0.0, np.zeros(p-2)]
    r = optimize.minimize(lambda t: -_ppo_ll(t, X, z, y).sum(), th0, method='BFGS', options=dict(gtol=1e-8, maxiter=5000))
    th = r.x; h = 1e-5; E = np.eye(p)
    S = np.array([(_ppo_ll(th+h*e, X, z, y)-_ppo_ll(th-h*e, X, z, y))/(2*h) for e in E]).T
    L = lambda t: _ppo_ll(t, X, z, y).sum()
    Hs = np.array([[(L(th+h*ei+h*ej)-L(th+h*ei-h*ej)-L(th-h*ei+h*ej)+L(th-h*ei-h*ej))/(4*h*h) for ej in E] for ei in E])
    Ai = np.linalg.inv(-(Hs+Hs.T)/2); groups = pd.unique(cl); G = len(groups)
    M = sum(np.outer(S[cl == g].sum(0), S[cl == g].sum(0)) for g in groups)
    V = Ai @ M @ Ai*G/(G-1); idx = 2+j
    return ci(th[idx], np.sqrt(V[idx, idx]), G-1)
