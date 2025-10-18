# cross-national-inference

**Ashkan Dashtban, PhD — Sept 2025**  
Cross-national healthcare cost inference using Bayesian models and MCMC — tools and workflows to estimate how new services (e.g., specialised Long-COVID services) change system-level cost burden. Implemented as part of the STIMULATE-ICP UK grant (~£6M).

---

## Project structure

```
cross-national-inference
├── bayesutils
│   ├── __init__.py
│   ├── diagnostics.py
│   └── plotting.py
├── License
├── main.ipynb
└── README.md
```

---

## Overview

This repository provides:

- A Python package `bayesutils` for Bayesian diagnostic analysis and plotting.  
- Jupyter notebook workflows (`main.ipynb`) demonstrating cross-national cost inference between **England** and **Wales**.  
- Reproducible implementations of Bayesian and Monte Carlo approaches for population-level and paired (pre→post) analyses.

Because national datasets cannot be linked at the individual level across countries, analyses rely on Bayesian inference and Monte Carlo simulation to:

1. Quantify cross-national differences in cost and utilisation rates.  
2. Assess whether differences are consistent with baseline population differences (**H₀**) or the introduction of specialised services (**H₁**).

---

## Quick start

```bash
# Clone repository
git clone https://github.com/AshkanDn/cross-national-inference.git
cd cross-national-inference

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install package in development mode
pip install --upgrade pip
pip install -e .
pip install numpy scipy pandas matplotlib jupyterlab arviz pymc3
```

Open and run the notebook:

```bash
jupyter lab
# open main.ipynb
```

---

## bayesutils package

**Submodules:**

- `diagnostics` — Bayesian p-values, posterior probabilities, tail probabilities, summary augmentation.  
- `plotting` — KDE, log-log CCDF, and grid overlay plotting.

**Convenience imports:**

```python
from bayesutils import diagnostics as diag
from bayesutils import plotting as plot
from bayesutils import bayesian_pvals, plot_kde
```

---

## Notebook workflow (`main.ipynb`)

**Section 1 — Cross-national population inference**  

- Hierarchical Bayesian Gamma model (partial pooling)  
- Parametric marginal Monte Carlo simulation  
- Robust Bayesian MCMC (Student-t log-cost)  
- Delta-method (lognormal pseudo-draws)

**Section 2 — Paired (pre→post) analysis**  

- Bayesian paired log-means model  
- Monte Carlo microdata reconstruction (Gaussian copula)  
- Paired Delta-method with pseudo-draws and KDE

---

## Best practices

- Pin dependency versions for reproducibility.  
- Set random seeds in simulation/MCMC code.  
- Use `nbdime` for notebook diffs:

```bash
pip install nbdime
nbdime config-git --enable --global
```

- Do not commit patient-identifiable data; use synthetic or de-identified datasets.  
- Include lightweight tests for `bayesutils` functions.

---

## License

Free for **educational, academic, and non-commercial research use**. Commercial use requires explicit permission from the author.

---

## Contribution

- Fork the repository  
- Create a feature branch  
- Add tests or examples  
- Open a pull request  

For major methodological changes, open an issue first to discuss.

---

## Citation / Contact

Ashkan Dashtban (2025). *cross-national-inference*. GitHub repository: [https://github.com/AshkanDn/cross-national-inference](https://github.com/AshkanDn/cross-national-inference)

**Email:** a.dashtban@ucl.ac.uk / dashtban.edu@gmail.com  
**GitHub:** [AshkanDn](https://github.com/AshkanDn)

---

## Example minimal reproducible workflow

```bash
git clone https://github.com/AshkanDn/cross-national-inference.git
cd cross-national-inference
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt   # if you add one
jupyter lab
# run main.ipynb
```

