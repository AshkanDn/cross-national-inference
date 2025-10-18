import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple, Union, Any
from tqdm import tqdm
from numpy.random import Generator




def empirical_pval_2sided(
    draws: Union[np.ndarray, list],
    null: float = 0.0,
    correction: bool = True
) -> float:
    """
    Compute empirical two-sided p-value from posterior draws versus a null value.
    
    Parameters
    ----------
    draws : array-like
        Posterior samples (1D array or list).
    null : float, optional
        Null hypothesis value to compare draws against, by default 0.0.
    correction : bool, optional
        Whether to apply small sample size correction to p-value (adds 1 to counts), by default True.
    
    Returns
    -------
    float
        Empirical two-sided p-value bounded between 0 and 1.
        Returns np.nan if input draws are empty after removing NaNs.
    """
    arr = np.asarray(draws)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    
    n = arr.size
    le = np.sum(arr <= null)
    ge = np.sum(arr >= null)
    
    if correction:
        p_le = (le + 1) / (n + 1)
        p_ge = (ge + 1) / (n + 1)
    else:
        p_le = le / n
        p_ge = ge / n
    
    p = 2.0 * min(p_le, p_ge)
    return float(min(max(p, 0.0), 1.0))



def _posterior_probabilities(draws, null=0.0, correction=True):
    """
    Compute posterior tail probabilities and empirical two-sided p-value for given posterior draws.
    
    Parameters:
    - draws: 1D array-like, posterior samples.
    - null: float, null value to calculate tail probabilities against.
    - correction: bool, whether to apply small sample correction when computing empirical p-value.
    
    Returns:
    - pd.DataFrame with one row containing:
      - P_gt: P(draws > null)
      - P_lt: P(draws < null)
      - p_empirical: Empirical two-sided p-value
      - null_value: The null value used
    """
    arr = np.asarray(draws)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return pd.DataFrame({
            "P_gt": [np.nan],
            "P_lt": [np.nan],
            "p_empirical": [np.nan],
            "null_value": [null]
        })

    # Tail probabilities
    P_gt = np.mean(arr > null)
    P_lt = np.mean(arr < null)

    # Empirical two-sided p-value
    n = arr.size
    le = np.sum(arr <= null)
    ge = np.sum(arr >= null)
    if correction:
        p_le = (le + 1) / (n + 1)
        p_ge = (ge + 1) / (n + 1)
    else:
        p_le = le / n
        p_ge = ge / n
    p_two_sided = 2.0 * min(p_le, p_ge)
    p_two_sided = min(max(p_two_sided, 0.0), 1.0)

    return pd.DataFrame({
        "P_gt": [float(P_gt)],
        "P_lt": [float(P_lt)],
        "p_empirical": [float(p_two_sided)],
        "null_value": [null]
    })


def posterior_probs(
    post_draws_map: Dict[str, Dict[str, Union[pd.Series, float, Any]]]
) -> pd.DataFrame:
    """
    Compute posterior tail probabilities and empirical p-values for each variable 
    from posterior draws and concatenate results into a single DataFrame.
    
    Parameters
    ----------
    post_draws_map : dict
        Dictionary mapping variable names to dictionaries containing:
        - 'draws': array-like posterior samples (e.g., numpy array, pandas Series)
        - 'null': float specifying null hypothesis value for tail probabilities
    
    Returns
    -------
    pd.DataFrame
        DataFrame indexed by variable names with columns:
        - P_gt: posterior probability draws > null
        - P_lt: posterior probability draws < null
        - p_empirical: two-sided empirical p-value
        - null_value: tested null hypothesis value
    
    Example
    -------
    >>> post_draws_map = {
    ...     "varA": {"draws": np.random.normal(0,1,1000), "null": 0.0},
    ...     "varB": {"draws": np.random.normal(1,1,1000), "null": 0.0},
    ... }
    >>> df = posterior_probs(post_draws_map)
    >>> print(df.head())
    """
    rows = []
    for varname, info in post_draws_map.items():
        draws = info["draws"]
        null = info["null"]
        df = _posterior_probabilities(draws, null=null)  # assumes _posterior_probabilities returns DataFrame with P_gt, P_lt, p_empirical
        df.index = [varname]
        rows.append(df)
    return pd.concat(rows)



def bayesian_pvals(
    draws_dict: Dict[str, Any],
    nulls: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compute Bayesian posterior tail probabilities for multiple variables.

    This function calculates the posterior probability that the posterior draws
    for each variable are greater or less than the specified null value,
    as well as a two-sided posterior tail probability p-value.

    Parameters
    ----------
    draws_dict : dict
        Dictionary mapping variable names to array-like posterior draws.
        Example: {"ratio": np.array([...]), "abs_DiD": np.array([...])}

    nulls : dict or None, optional
        Dictionary of null values for each variable.
        If None, defaults are:
          - 1.0 for variables not containing 'abs' in their names.
          - 0.0 for variables containing 'abs' in their names.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per variable containing columns:
        - variable: str, variable name
        - null_value: float, null hypothesis value for the variable
        - P_gt_null: float, posterior probability that draws > null
        - P_lt_null: float, posterior probability that draws < null
        - p_two_sided: float, two-sided posterior tail probability p-value

    Examples
    --------
    >>> draws_dict = {
    ...     "ratio": np.random.normal(1, 0.1, 1000),
    ...     "abs_DiD": np.random.normal(0, 0.05, 1000)
    ... }
    >>> nulls = {"ratio": 1.0, "abs_DiD": 0.0}
    >>> df = bayesian_pvals(draws_dict, nulls=nulls)
    >>> print(df)
    """
    results = []
    if nulls is None:
        nulls = {}

    for varname, arr in draws_dict.items():
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]

        if "abs" in varname.lower():
            null_val = nulls.get(varname, 0.0)
        else:
            null_val = nulls.get(varname, 1.0)

        P_gt = np.mean(arr > null_val)
        P_lt = np.mean(arr < null_val)
        p_two_sided = 2 * min(P_gt, P_lt)

        results.append({
            "variable": varname,
            "null_value": null_val,
            "P_gt_null": float(P_gt),
            "P_lt_null": float(P_lt),
            "p_two_sided": float(p_two_sided)
        })

    return pd.DataFrame(results)






def posterior_tail_probs(
    draws: Union[np.ndarray, list],
    null: float = 0.0
) -> Dict[str, float]:
    """
    Compute posterior tail probabilities and empirical two-sided p-value from posterior draws.

    Parameters
    ----------
    draws : array-like (np.ndarray or list)
        Posterior samples. Should be a 1-D array or list of numeric draws.
    null : float, optional
        Null hypothesis value to compare draws against (default is 0.0).

    Returns
    -------
    dict
        Dictionary containing:
        - "P_gt": posterior probability that draws > null
        - "P_lt": posterior probability that draws < null
        - "p_emp": empirical two-sided p-value

    Notes
    -----
    - NaN values in draws will be automatically removed.
    - Returns NaN values for all keys if the input draws are empty after removing NaNs.
    """
    arr = np.asarray(draws)
    arr = arr[~np.isnan(arr)]

    if arr.size == 0:
        return {"P_gt": np.nan, "P_lt": np.nan, "p_emp": np.nan}

    P_gt = np.mean(arr > null)
    P_lt = np.mean(arr < null)
    p_emp = empirical_pval_2sided(arr, null=null, correction=True)

    return {"P_gt": float(P_gt), "P_lt": float(P_lt), "p_emp": float(p_emp)}


def augment_empirical_pvals(
    post_draws_map: Dict[str, Dict[str, Any]],
    summary_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Augment an ArviZ summary DataFrame with empirical posterior tail probabilities and p-values.
    
    Parameters
    ----------
    post_draws_map : dict
        Dictionary mapping variable names to dictionaries containing:
        - "draws": array-like posterior samples (e.g. numpy array, pandas Series)
        - "null": float, the null hypothesis value to test against
    
    Returns
    -------
    pandas.DataFrame
        A copy of the input summary_table with added columns:
        - "p_empirical": empirical two-sided p-value
        - "P_gt_null": posterior probability draws > null
        - "P_lt_null": posterior probability draws < null
        - "null_value": the null hypothesis value tested
    
    Notes
    -----
    If a variable from post_draws_map is not present in the summary_table index,
    a new row will be created for it.
    
    Examples
    --------
    >>> post_draws_map = {
    ...     "fold_W": {"draws": np.random.normal(1, 0.1, 1000), "null": 1.0},
    ...     "fold_E": {"draws": np.random.normal(1, 0.1, 1000), "null": 1.0},
    ... }

    # OR:
    >>> post_draws_map = {
            "fold_W": {"draws": post_fold_w, "null": 1.0},
            "fold_E": {"draws": post_fold_e, "null": 1.0},
            "ratio" : {"draws": post_ratio,  "null": 1.0},
            "abs_DiD": {"draws": post_abs,   "null": 0.0}
        }
            
    >>> summary = arviz.summary(inference_data)
    >>> augmented_summary = augment_empirical_pvals(post_draws_map, summary)
    >>> display(augmented_summary.head())
    """
    aug_summary = summary_table.copy()
    
    for varname, info in post_draws_map.items():
        draws = info["draws"]
        null = info["null"]
        
        probs = posterior_tail_probs(draws, null=null)  # assumes this function exists
        
        # Create a new row filled with NaNs if missing (rare)
        if varname not in aug_summary.index:
            aug_summary.loc[varname] = np.nan
        
        aug_summary.loc[varname, "p_empirical"] = probs["p_emp"]
        aug_summary.loc[varname, "P_gt_null"] = probs["P_gt"]
        aug_summary.loc[varname, "P_lt_null"] = probs["P_lt"]
        aug_summary.loc[varname, "null_value"] = null
    
    return aug_summary



def summarise_array(x: np.ndarray, ci_level: float = 95) -> Dict[str, Any]:
    """
    Summarise an array by computing mean, standard deviation, and two-sided percentile confidence interval.

    Parameters
    ----------
    x : np.ndarray
        Numeric input array to summarize.
    ci_level : float, optional
        Confidence interval level as a percentage (default is 95).

    Returns
    -------
    dict
        Dictionary with keys:
        - "mean": float, mean of finite values in `x`
        - "sd": float, sample standard deviation (ddof=1)
        - "ci_low": float, lower bound of percentile CI
        - "ci_high": float, upper bound of percentile CI
        NaN if input is empty or contains no finite values.
    """
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "sd": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    alpha = 100 - ci_level
    lower_pct = alpha / 2
    upper_pct = 100 - lower_pct

    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "ci_low": float(np.percentile(x, lower_pct)),
        "ci_high": float(np.percentile(x, upper_pct))
    }


def gamma_params_from_mean_sd(mean: float, sd: float) -> (float, float):
    """
    Convert mean and SD to Gamma shape and scale parameters.
    """
    shape = (mean / sd) ** 2
    scale = (sd ** 2) / mean
    return shape, scale

def sim_indiv_from_summary(
    mean_obs: float,
    sd_obs: float,
    n: int,
    rr: Generator,
    marginal: str = 'lognormal',
    size_override: Optional[int] = None
) -> np.ndarray:
    """
    Simulate individual-level samples from marginal summaries using specified distribution.

    Parameters
    ----------
    mean_obs : float
        Observed mean of the marginal distribution.
    sd_obs : float
        Observed standard deviation of the marginal distribution.
    n : int
        Number of individuals to simulate.
    rr : numpy.random.Generator
        Random number generator instance.
    marginal : str, optional
        Marginal distribution type, either 'lognormal' or 'gamma'.
    size_override : int or None, optional
        If provided, overrides `n` for number of samples to generate.

    Returns
    -------
    np.ndarray
        Array of simulated individual samples.
    """
    m = max(mean_obs, 1e-12)
    s = max(sd_obs, 1e-12)
    n_out = size_override if size_override is not None else n

    if marginal == 'lognormal':
        sigma2 = math.log1p((s ** 2) / (m ** 2))
        sigma = math.sqrt(max(sigma2, 0.0))
        mu_log = math.log(m) - 0.5 * sigma2
        return rr.lognormal(mean=mu_log, sigma=sigma, size=n_out)
    elif marginal == 'gamma':
        a, th = gamma_params_from_mean_sd(m, s)
        return rr.gamma(shape=a, scale=th, size=n_out)
    else:
        raise ValueError("marginal must be 'lognormal' or 'gamma'")


def empirical_lognormal_sigma_from_sample(mean_obs, sd_obs):
    m = max(mean_obs, 1e-12)
    s = max(sd_obs, 1e-12)
    cv2 = (s/m)**2
    sigma2 = math.log1p(cv2)
    return math.sqrt(max(sigma2, 0.0))

def lognormal_params_from_mean_sd(mean, sd):
    mean = max(mean, 1e-12)
    var = max(sd**2, 1e-12)
    sigma2 = math.log1p(var / (mean**2))
    sigma = math.sqrt(max(sigma2, 0.0))
    mu = math.log(mean) - 0.5 * sigma2
    return mu, sigma

from typing import Optional, Tuple
import math
import numpy as np
import scipy.stats as stats

def sim_paired_from_summaries(
    mean_pre: float,
    sd_pre: float,
    mean_post: float,
    sd_post: float,
    n: int,
    rho: float = 0.15,
    rng: Optional[object] = None,
    marginal: str = "lognormal",
    size_override: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate paired pre/post microdata from marginal summaries using a Gaussian copula.

    Parameters
    ----------
    mean_pre, sd_pre : float
        Observed sample mean and sd for the 'pre' marginal.
    mean_post, sd_post : float
        Observed sample mean and sd for the 'post' marginal.
    n : int
        Original reported sample size (kept for API compatibility; not strictly required
        for marginal parameter estimation here).
    rho : float, default 0.15
        Correlation between the *log-marginals* (for lognormal) or the latent normals used by the copula.
        Must be between -1 and 1.
    rng : np.random.Generator or int or None
        RNG object or seed. If int, a Generator is created from that seed.
        If None, a new non-deterministic Generator is used.
    marginal : {"lognormal","gamma"}, default "lognormal"
        Marginal family to simulate from. "lognormal" exponentiates correlated normals;
        "gamma" uses Gaussian copula (normal CDF → Uniform → gamma.ppf).
    size_override : int or None
        Number of paired samples to generate. If None, defaults to 'n' (or 1 if n<=0).

    Returns
    -------
    (pre_samples, post_samples) : Tuple[np.ndarray, np.ndarray]
        Two 1-d float arrays of length `size_override` containing paired simulated observations.
    """
    # -----------------------
    # Input sanitization
    # -----------------------
    if size_override is None:
        size = max(1, int(n or 1))
    else:
        size = max(1, int(size_override))

    # RNG handling
    if rng is None:
        rng_gen = np.random.default_rng()
    elif isinstance(rng, np.random.Generator):
        rng_gen = rng
    else:
        # assume integer seed
        rng_gen = np.random.default_rng(int(rng))

    # clamp correlation to valid range
    rho = float(rho)
    if rho >= 1.0:
        rho = 0.999999
    if rho <= -1.0:
        rho = -0.999999

    # protect means and sds
    mean_pre = float(max(mean_pre, 1e-12))
    mean_post = float(max(mean_post, 1e-12))
    sd_pre = float(max(sd_pre, 1e-12))
    sd_post = float(max(sd_post, 1e-12))

    # Cholesky for 2x2 correlation matrix [[1,rho],[rho,1]]
    # L = [[1,0],[rho, sqrt(1-rho^2)]] gives L @ L.T = [[1,rho],[rho,1]]
    chol = np.array([[1.0, 0.0], [rho, math.sqrt(max(0.0, 1.0 - rho ** 2))]], dtype=float)

    # draw independent standard normals and apply Cholesky to produce correlated normals
    z = rng_gen.standard_normal(size=(2, size))
    correlated = chol @ z  # shape (2, size)
    z1 = correlated[0, :]
    z2 = correlated[1, :]

    # For lognormal: build correlated normals on log-scale and exponentiate
    if marginal.lower() in ("lognormal", "log-normal", "ln"):
        # compute lognormal parameters (mu_log, sigma_log) from mean & sd
        def _log_params(m: float, s: float) -> Tuple[float, float]:
            cv2 = (s / m) ** 2
            sigma2 = math.log1p(cv2)
            sigma = math.sqrt(max(sigma2, 0.0))
            mu = math.log(m) - 0.5 * sigma2
            return mu, sigma

        mu_pre, sigma_pre = _log_params(mean_pre, sd_pre)
        mu_post, sigma_post = _log_params(mean_post, sd_post)

        # correlated normals on log-scale: Y = mu + sigma * Z
        y_pre = mu_pre + sigma_pre * z1
        y_post = mu_post + sigma_post * z2

        pre_samples = np.exp(y_pre)
        post_samples = np.exp(y_post)
        return pre_samples.astype(float), post_samples.astype(float)

    # For gamma: use Gaussian copula -> uniforms -> gamma.ppf
    elif marginal.lower() in ("gamma",):
        # estimate alpha (shape) and theta (scale) from mean & sd
        def _gamma_params(m: float, s: float) -> Tuple[float, float]:
            var = max(s**2, 1e-12)
            alpha = max((m * m) / var, 1e-6)   # shape
            theta = var / m if m > 0 else 1.0  # scale
            return alpha, theta

        a_pre, scale_pre = _gamma_params(mean_pre, sd_pre)
        a_post, scale_post = _gamma_params(mean_post, sd_post)

        # convert correlated normals to uniform via standard normal CDF
        eps = 1e-12
        u_pre = stats.norm.cdf(z1)
        u_post = stats.norm.cdf(z2)
        # clamp to avoid 0/1 exactly
        u_pre = np.clip(u_pre, eps, 1 - eps)
        u_post = np.clip(u_post, eps, 1 - eps)

        # invert gamma CDF
        pre_samples = stats.gamma.ppf(u_pre, a=a_pre, scale=scale_pre)
        post_samples = stats.gamma.ppf(u_post, a=a_post, scale=scale_post)

        # gamma.ppf may produce NaNs for extreme params; guard by fallback to random gamma draws
        # (should be rare with clamps)
        if np.any(~np.isfinite(pre_samples)):
            pre_samples = rng_gen.gamma(shape=a_pre, scale=scale_pre, size=size)
        if np.any(~np.isfinite(post_samples)):
            post_samples = rng_gen.gamma(shape=a_post, scale=scale_post, size=size)

        return pre_samples.astype(float), post_samples.astype(float)

    else:
        raise ValueError(f"Unsupported marginal: {marginal!r}. Supported: 'lognormal', 'gamma'.")
