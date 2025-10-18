
"""
bayesutils package

This package contains Bayesian diagnostic analysis and plotting utilities.

Created by Ashkan Dashtban for non-profit, educational, and research purposes.
Commercial use is prohibited without explicit permission.

Contact:
    Email: dashtban.edu@gmail.com
    Phone: +44 07788 228085
    Address: Reading, UK

Submodules
----------
- diagnostics: Bayesian p-value, posterior probabilities, tail probs, and summary augmentation.
- plotting: KDE and log-log CCDF plot functions.


License
-------
This software may be used freely for educational, academic, and non-commercial research purposes 
under the terms of the accompanying license. Commercial or for-profit use requires prior authorization
from the author.
"""


# Import submodules : eg., from bayesutils import diagnostics as diag
from . import diagnostics
from . import plotting

# Import key functions for convenience directly from package level
from .diagnostics import (
    bayesian_pvals,
    posterior_probs,
    posterior_tail_probs,
    augment_empirical_pvals,
    summarise_array,
    empirical_pval_2sided,
    gamma_params_from_mean_sd,
    sim_indiv_from_summary,
    empirical_lognormal_sigma_from_sample,
    lognormal_params_from_mean_sd,
    sim_paired_from_summaries,

)

from .plotting import (
    plot_kde, 
    plot_loglog_ccdf, 
    plot_kde_grid,
    plot_kde_grid_overlay
)

__all__ = [
    "diagnostics",
    "plotting",
    "bayesian_pvals",
    "posterior_probs",
    "posterior_tail_probs",
    "augment_empirical_pvals",
    "plot_kde",
    "plot_loglog_ccdf",
    "summarise_array",
    "empirical_pval_2sided",
    "gamma_params_from_mean_sd",
    "sim_indiv_from_summary",
    "empirical_lognormal_sigma_from_sample",
    "lognormal_params_from_mean_sd",
    "plot_kde_grid",
    "plot_kde_grid_overlay",
    "sim_paired_from_summaries",
    
]

