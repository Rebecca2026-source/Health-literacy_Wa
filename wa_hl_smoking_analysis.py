"""wa_hl_smoking_analysis.py

Health literacy, smoking status and nicotine dependence among Wa adults in rural southwest China.
Complete analysis code for the revised manuscript (PLOS ONE, PONE-D-26-31483).

Usage:
    python wa_hl_smoking_analysis.py <path to analytic data .xlsx> [output folder]

Requirements: Python >= 3.10, numpy, pandas, scipy, matplotlib, openpyxl.
All statistical methods are implemented in cr2_methods.py (same folder).

Outputs (in the output folder):
    results.json   all estimates used in the manuscript, tables and Supporting Information
    results.txt    readable log of the same results
    S3_Fig.png     restricted cubic spline curves for health literacy
"""
import sys, os, json, warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cr2_methods import *

DATA = sys.argv[1] if len(sys.argv) > 1 else 'clean_data_final_analysis_680_village_std.xlsx'
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(OUTDIR, exist_ok=True)
SEED = 20261102
R, LOG = {}, []

def log(*a):
    s = ' '.join(str(x) for x in a); print(s); LOG.append(s)

def fmt(r, d=3):
    df = '' if np.isinf(r.get('df', np.inf)) else f"; df={r['df']:.1f}"
    return f"{r['est']:.{d}f} ({r['lo']:.{d}f} to {r['hi']:.{d}f}); P={r['p']:.3f}{df}"

def keep(key, r):
    R[key] = {k: (float(v) if v is not None else None) for k, v in r.items()}
    log(f'{key}: {fmt(r, 3 if "linear" not in key else 2)}')
    return r

# =====================================================================
# 1. Data preparation
# =====================================================================
d = pd.read_excel(DATA)
d['village_name'] = d.village_text.str.extract(r'县(.+?村)')[0]
assert d.village_name.notna().all() and d.village_name.nunique() == 8, 'village coding'
# anonymised village labels A-H (by sample size), used in all outputs
order = d.village_name.value_counts().index.tolist()
d['village'] = d.village_name.map({v: 'Village ' + 'ABCDEFGH'[i] for i, v in enumerate(order)})

d.loc[d.family_member == 32, 'family_member'] = 5           # data-entry error verified against questionnaire
SH = ['gulvxiyan_age', 'muqianyanling_month', 'jinyiyue_meitianpingjunxiyan']
for v in SH:                                                  # negative codes = missing
    d.loc[d[v] < 0, v] = np.nan

# exposure: total health-literacy score (0-66) = sum of the three aspect scores
d['hl'] = d[['Basic_Knowledge_Concepts_22_28', 'Healthy_Lifestyle_and_Behavior_18_22', 'Health_Skills_13_16']].sum(axis=1)
d['hl5'] = d.hl/5                                             # per 5-point increment
d['hl5_noBB05'] = (d.hl-d.BB05)/5                             # item BB05 (smoking harms) removed

# covariates
d['male'] = (d.gender == 1)*1.
d['edu_prim'] = (d.education == 2)*1.                         # reference: illiterate
d['edu_mid'] = (d.education >= 3)*1.                          # middle school or above
d['loginc'] = np.log(d.family_income)
d['farmer'] = (d.Occupation == 6)*1.
d['married'] = (d.marriage == 2)*1.
d['drink'] = (d.AUDIT1 >= 3)*1.                               # at least 2-4 times per month
d['chronic'] = (d.chronic_disease != -3)*1.
d['hhsize'] = d.family_member.astype(float)

# outcomes. smoking_status_raw: 1 daily, 2 non-daily, 3 former, 4 never (question A10)
d['ever'] = d.smoking_status_raw.isin([1, 2, 3])*1.
d['current'] = d.smoking_status_raw.isin([1, 2])*1.
d['daily'] = (d.smoking_status_raw == 1)*1.
d['ftnd_sev'] = np.select([d.FTND_total <= 3, d.FTND_total <= 6], [0, 1], 2)   # low / moderate / high

M1 = ['age', 'male']
M2 = M1+['edu_prim', 'edu_mid']
CORE = M2+['loginc', 'farmer', 'married']                     # main model (confounders)
EXT = CORE+['drink', 'BMI', 'PHQ_total', 'GAD_total', 'chronic']   # extended model (former M4)

ever = d.copy(); ever['y'] = ever.ever
cur = d[d.ever == 1].copy(); cur['y'] = cur.current
cs = d[d.current == 1].copy()

def Xmat(dd, cols, fe=False, exp='hl5'):
    X = dd[[exp]+cols].astype(float)
    if fe: X = pd.concat([X, pd.get_dummies(dd.village, drop_first=True).astype(float)], axis=1)
    X.insert(0, 'const', 1.); return X.values

def e_j(k, j=1):
    c = np.zeros(k); c[j] = 1.; return c

def binary(dd, cols, fam='poisson', method='cr2', fe=False, exp='hl5', pw=None):
    X = Xmat(dd, cols, fe, exp)
    return glm_contrast(dd.y.values, X, dd.village.values, fam, e_j(X.shape[1]), method, pw)

def ordX(df, cols, exp='hl5', fe=False):
    X = df[[exp]+cols].astype(float)
    if fe: X = pd.concat([X, pd.get_dummies(df.village, drop_first=True).astype(float)], axis=1)
    X = X.values; return X-X.mean(0)

# =====================================================================
# 2. Sample description
# =====================================================================
log('=== Sample ===')
R['n'] = dict(total=len(d), never=int((d.ever == 0).sum()), former=int(((d.ever == 1) & (d.current == 0)).sum()),
              current=len(cs), daily=int(d.daily.sum()), nondaily=int((d.smoking_status_raw == 2).sum()),
              ftnd_low=int((cs.ftnd_sev == 0).sum()), ftnd_mod=int((cs.ftnd_sev == 1).sum()), ftnd_high=int((cs.ftnd_sev == 2).sum()),
              smoking_history_available=int(cs.gulvxiyan_age.notna().sum()))
R['village_n'] = d.village.value_counts().sort_index().to_dict()
R['former_by_village'] = {k: int(v) for k, v in cur.groupby('village').y.apply(lambda s: (s == 0).sum()).items()}
R['icc'] = dict(hl=icc_anova(d.hl, d.village), ever=icc_anova(ever.y, ever.village),
                current=icc_anova(cur.y, cur.village), ftnd=icc_anova(cs.FTND_total, cs.village))
o = cs.dropna(subset=SH)
py = o.jinyiyue_meitianpingjunxiyan/20*o.muqianyanling_month/12
R['smoking_history'] = dict(init_mean=o.gulvxiyan_age.mean(), init_sd=o.gulvxiyan_age.std(),
                            dur_mean=(o.muqianyanling_month/12).mean(), dur_sd=(o.muqianyanling_month/12).std(),
                            cpd_mean=o.jinyiyue_meitianpingjunxiyan.mean(), cpd_sd=o.jinyiyue_meitianpingjunxiyan.std(),
                            py_median=py.median(), py_q1=py.quantile(.25), py_q3=py.quantile(.75))
for k in ['n', 'village_n', 'former_by_village', 'icc', 'smoking_history']: log(k, R[k])

# Table 2: Spearman correlations (unchanged from the original analysis)
preds = {'Total health-literacy score': 'hl', 'Knowledge and attitudes': 'Basic_Knowledge_Concepts_22_28',
         'Lifestyle and behavior': 'Healthy_Lifestyle_and_Behavior_18_22', 'Health-related skills': 'Health_Skills_13_16'}
R['spearman'] = {}
for lab, v in preds.items():
    r1 = stats.spearmanr(ever[v], ever.y); r2 = stats.spearmanr(cur[v], cur.y)
    R['spearman'][lab] = [r1.statistic, r1.pvalue, r2.statistic, r2.pvalue]
log('spearman', {k: np.round(v, 3).tolist() for k, v in R['spearman'].items()})

# =====================================================================
# 3. Main models (Table 3): modified Poisson (aPR) and logistic (OR),
#    village-clustered CR2 variance with Satterthwaite degrees of freedom
# =====================================================================
log('\n=== Table 3 ===')
for lab, dd in [('ever', ever), ('current', cur)]:
    for mn, cols in [('M1', M1), ('M2', M2), ('core', CORE), ('extended', EXT)]:
        keep(f'T3_{lab}_{mn}_aPR', binary(dd, cols))
        keep(f'T3_{lab}_{mn}_OR', binary(dd, cols, 'logit'))
    keep(f'T3_{lab}_originalM4_HC0_aPR', binary(dd, EXT, method='hc0'))   # reproduces the original manuscript

# =====================================================================
# 4. Sensitivity analyses for clustering (S1 Table)
# =====================================================================
log('\n=== S1 Table: variance estimators and village structure ===')
for lab, dd in [('ever', ever), ('current', cur)]:
    keep(f'S1_{lab}_HC0', binary(dd, CORE, method='hc0'))
    keep(f'S1_{lab}_CR1', binary(dd, CORE, method='cr1'))
    keep(f'S1_{lab}_CR2', binary(dd, CORE))
    keep(f'S1_{lab}_MD', binary(dd, CORE, method='md'))
    keep(f'S1_{lab}_FE', binary(dd, CORE, fe=True))
    th, V = ri_logit(dd.y.values, Xmat(dd, CORE), dd.village.values)
    keep(f'S1_{lab}_RI_OR', ci(th[1], np.sqrt(V[1, 1]), np.inf)); R[f'S1_{lab}_RI_sd'] = float(np.exp(th[-1]))
    keep(f'S1_{lab}_CR2_OR', binary(dd, CORE, 'logit'))
    loo = {}
    for v in sorted(dd.village.unique()):
        loo[v] = binary(dd[dd.village != v], CORE)
        log(f'   leave out {v}: {fmt(loo[v])}')
    R[f'S4_{lab}_LOO'] = {k: {a: float(b) for a, b in v.items()} for k, v in loo.items()}

# =====================================================================
# 5. Further sensitivity analyses (S4 Table)
# =====================================================================
log('\n=== S4 Table ===')
for lab, dd in [('ever', ever), ('current', cur)]:
    keep(f'S4_{lab}_hhweight', binary(dd, CORE, pw=dd.hhsize.values/dd.hhsize.mean()))
    keep(f'S4_{lab}_hhadjust', binary(dd, CORE+['hhsize']))
    dl = dd[dd.smoking_status_raw != 2]
    keep(f'S4_{lab}_dailyonly', binary(dl, CORE)); R[f'S4_{lab}_dailyonly']['n'] = len(dl)
    keep(f'S2_{lab}_noBB05', binary(dd, CORE, exp='hl5_noBB05'))
    # sex interaction
    dd2 = dd.assign(hlxmale=dd.hl5*dd.male)
    X = Xmat(dd2, ['hlxmale']+CORE); b, Xt, rt = fit_glm(dd2.y.values, X, 'poisson'); k = X.shape[1]
    keep(f'S4_{lab}_int_p', ci(*cr2(b, Xt, rt, dd2.village.values, e_j(k, 2))))
    keep(f'S4_{lab}_women', ci(*cr2(b, Xt, rt, dd2.village.values, e_j(k, 1))))
    keep(f'S4_{lab}_men', ci(*cr2(b, Xt, rt, dd2.village.values, e_j(k, 1)+e_j(k, 2))))
    # linearity: restricted cubic spline, knots at 10th/50th/90th percentiles
    kn = np.percentile(dd.hl, [10, 50, 90]); S = rcs_basis(dd.hl.values, kn)/5
    base = Xmat(dd, CORE); X = np.column_stack([base[:, :2], S, base[:, 2:]])
    keep(f'S4_{lab}_nonlinear', glm_contrast(dd.y.values, X, dd.village.values, 'poisson', e_j(X.shape[1], 2)))
    R[f'S4_{lab}_knots'] = kn.tolist()
    q = pd.qcut(dd.hl, 4, labels=False); cuts = pd.qcut(dd.hl, 4).cat.categories
    Xq = np.column_stack([np.ones(len(dd)), (q == 1)*1., (q == 2)*1., (q == 3)*1., dd[CORE].values.astype(float)])
    R[f'S4_{lab}_quartile_cuts'] = [str(c) for c in cuts]
    for j in (1, 2, 3):
        keep(f'S4_{lab}_Q{j+1}', glm_contrast(dd.y.values, Xq, dd.village.values, 'poisson', e_j(Xq.shape[1], j)))

# Firth penalised logistic regression, current vs former (profile penalised-likelihood CI)
for mn, cols, fe in [('core', CORE, False), ('extended', EXT, False), ('core_FE', CORE, True)]:
    keep(f'S4_current_firth_{mn}', firth_ci(cur.y.values, Xmat(cur, cols, fe), 1))

# =====================================================================
# 6. FTND severity (Table 4, S3 Table)
# =====================================================================
log('\n=== Table 4 / S3 Table ===')
y, g = cs.ftnd_sev.values, cs.village.values
keep('T4_core', ordinal_contrast(ordX(cs, CORE), y, g, 0, 'cr1'))
keep('T4_extended', ordinal_contrast(ordX(cs, EXT), y, g, 0, 'cr1'))
keep('T4_originalM4_HC0', ordinal_contrast(ordX(cs, EXT), y, g, 0, 'hc0'))
chi, dfo, pb, per, bs = brant(ordX(cs, CORE), y)
R['T4_brant'] = dict(chi2=chi, df=dfo, p=pb, p_vars=dict(zip(['hl5']+CORE, per)))
log('Brant core', R['T4_brant'])
Xp = ordX(cs, [c for c in CORE if c != 'age']); zp = (cs.age-cs.age.mean()).values
keep('T4_partialPO_age', ppo_contrast(Xp, zp, y, g, 0))
keep('S3_FE', ordinal_contrast(ordX(cs, CORE, fe=True), y, g, 0, 'cr1'))
keep('S3_noBB05', ordinal_contrast(ordX(cs, CORE, 'hl5_noBB05'), y, g, 0, 'cr1'))
(th, V), K = ri_ordinal(y, ordX(cs, CORE), g)
keep('S3_RI', ci(th[K-1], np.sqrt(V[K-1, K-1]), np.inf)); R['S3_RI_sd'] = float(np.exp(th[-1]))
cs2 = cs.assign(y=cs.FTND_total.astype(float))
for mn, cols, fe in [('core', CORE, False), ('extended', EXT, False), ('core_FE', CORE, True)]:
    X = Xmat(cs2, cols, fe)
    keep(f'S3_linear_{mn}', glm_contrast(cs2.y.values, X, g, 'linear', e_j(X.shape[1]), 'cr2', expo=False))

# ---- smoking-history sensitivity models with multiple imputation (FCS, predictive mean matching) ----
c0 = cs.reset_index(drop=True); miss = c0[SH].isna().any(axis=1).values
VD = pd.get_dummies(c0.village, drop_first=True).astype(float)
BASE = pd.concat([c0[['age', 'male', 'edu_prim', 'edu_mid', 'loginc', 'farmer', 'married', 'hl', 'FTND_total', 'daily']].astype(float), VD], axis=1)

def fcs_pmm(rng, iters=20, k=5):
    df = c0.copy()
    for v in SH: df.loc[df[v].isna(), v] = rng.choice(c0[v].dropna().values, df[v].isna().sum())
    for _ in range(iters):
        for v in SH:
            mv = c0[v].isna().values
            X = np.column_stack([np.ones(len(df)), BASE.values, df[[o for o in SH if o != v]].values])
            Xo, yo, Xm = X[~mv], df.loc[~mv, v].values.astype(float), X[mv]
            XtX = Xo.T @ Xo; bh = np.linalg.solve(XtX, Xo.T @ yo)
            s2 = np.sum((yo-Xo @ bh)**2)/rng.chisquare(len(yo)-X.shape[1])
            bs_ = rng.multivariate_normal(bh, s2*np.linalg.inv(XtX))
            po, pm = Xo @ bh, Xm @ bs_
            df.loc[mv, v] = [yo[rng.choice(np.argsort(np.abs(po-p))[:k])] for p in pm]
    return df

def ordinal_est_var(df, cols):
    X = ordX(df, cols); th, Ai, S, K = fit_ordinal(X, df.ftnd_sev.values)
    gg = df.village.values; G = len(np.unique(gg))
    M = sum(np.outer(S[gg == u].sum(0), S[gg == u].sum(0)) for u in np.unique(gg))
    V = Ai @ M @ Ai*G/(G-1); return th[K-1], V[K-1, K-1]

def rubin(est, var, dfcom=7):
    m = len(est); q = np.mean(est); W = np.mean(var); B = np.var(est, ddof=1); T = W+(1+1/m)*B
    lam = (1+1/m)*B/T; dfold = (m-1)/lam**2 if lam > 0 else np.inf
    dfobs = (dfcom+1)/(dfcom+3)*dfcom*(1-lam); dfa = 1/(1/dfold+1/dfobs) if np.isfinite(dfold) else dfobs
    t = stats.t.ppf(.975, dfa); se = np.sqrt(T)
    return dict(est=np.exp(q), lo=np.exp(q-t*se), hi=np.exp(q+t*se), p=2*stats.t.sf(abs(q/se), dfa), df=dfa)

rng = np.random.default_rng(SEED); M_IMP = 50
est = {v: [] for v in SH[:2]}; var = {v: [] for v in SH[:2]}
for _ in range(M_IMP):
    imp = fcs_pmm(rng)
    for v in SH[:2]:
        e, vv = ordinal_est_var(imp, CORE+[v]); est[v].append(e); var[v].append(vv)
keep('T4_initage_MI', rubin(est['gulvxiyan_age'], var['gulvxiyan_age']))
keep('T4_duration_MI', rubin(est['muqianyanling_month'], var['muqianyanling_month']))
for v, key in [('gulvxiyan_age', 'S3_initage_CC'), ('muqianyanling_month', 'S3_duration_CC')]:
    sub = cs.dropna(subset=[v]); keep(key, ordinal_contrast(ordX(sub, CORE+[v]), sub.ftnd_sev.values, sub.village.values, 0, 'cr1'))

# =====================================================================
# 7. Within- and between-village decomposition (S5 Table)
# =====================================================================
log('\n=== S5 Table ===')
for lab, dd, fam in [('ever', ever, 'poisson'), ('current', cur, 'poisson'), ('ftnd_linear', cs2, 'linear')]:
    dd2 = dd.copy(); dd2['hl_b'] = dd2.groupby('village').hl5.transform('mean'); dd2['hl_w'] = dd2.hl5-dd2.hl_b
    X = np.column_stack([np.ones(len(dd2)), dd2[['hl_w', 'hl_b']+CORE].values.astype(float)])
    ex = fam != 'linear'
    keep(f'S5_{lab}_within', glm_contrast(dd2.y.values, X, dd2.village.values, fam, e_j(X.shape[1], 1), expo=ex))
    keep(f'S5_{lab}_between', glm_contrast(dd2.y.values, X, dd2.village.values, fam, e_j(X.shape[1], 2), expo=ex))

# =====================================================================
# 8. S3 Fig: spline curves (reference = median score)
# =====================================================================
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), dpi=300)
for ax, (lab, dd, title) in zip(axes, [('ever', ever, 'A. Ever vs never smoking'), ('current', cur, 'B. Current vs former smoking (ever smokers)')]):
    kn = np.percentile(dd.hl, [10, 50, 90]); S = rcs_basis(dd.hl.values, kn)
    base = Xmat(dd, CORE); X = np.column_stack([base[:, :1], dd.hl.values, S, base[:, 2:]])
    b, Xt, rt = fit_glm(dd.y.values, X, 'poisson')
    grid = np.linspace(np.percentile(dd.hl, 2.5), np.percentile(dd.hl, 97.5), 40); ref = np.median(dd.hl)
    Sg = rcs_basis(grid, kn); Sr = rcs_basis(np.array([ref]), kn)
    lo, mid, hi = [], [], []
    for gi, sg in zip(grid, Sg):
        c = np.zeros(X.shape[1]); c[1] = gi-ref; c[2:2+S.shape[1]] = sg-Sr[0]
        r = ci(*cr2(b, Xt, rt, dd.village.values, c)) if abs(gi-ref) > 1e-9 else dict(est=1, lo=1, hi=1)
        mid.append(r['est']); lo.append(r['lo']); hi.append(r['hi'])
    ax.fill_between(grid, lo, hi, color='0.8', lw=0); ax.plot(grid, mid, color='k', lw=1.5)
    ax.axhline(1, color='0.4', lw=0.8, ls='--'); ax.set_yscale('log')
    ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]
    ax.set_yticks(ticks); ax.set_yticklabels([f'{t:.1f}' for t in ticks]); ax.minorticks_off()
    ax.set_xlabel('Total health-literacy score'); ax.set_ylabel('Adjusted prevalence ratio'); ax.set_title(title, fontsize=9, loc='left')
    ax.plot(dd.hl, np.full(len(dd), 0.62), '|', color='0.5', ms=4, alpha=.4)   # observed scores
    ax.set_ylim(0.6, 1.5)
plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'S3_Fig.png')); plt.close()

json.dump(R, open(os.path.join(OUTDIR, 'results.json'), 'w'), indent=1, ensure_ascii=False, default=float)
open(os.path.join(OUTDIR, 'results.txt'), 'w').write('\n'.join(LOG))
log('\nDone.')
