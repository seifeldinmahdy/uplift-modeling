# Uplift Modeling: Targeting Who Is *Causally Moved* by an Intervention

## The problem in one line
Response models predict who will convert. Uplift models predict **who converts *because of* the intervention** — a fundamentally different (and more valuable) question.

## Why uplift beats response modeling
A customer who buys without a promo wastes your budget. A customer who buys *only because* of a promo is your true target. Treating high-responders who would have converted anyway is zero ROI; treating "do-not-disturb" customers is negative ROI. Uplift modeling separates these groups by estimating the **Conditional Average Treatment Effect (CATE)**: τ(x) = E[Y(1) - Y(0) | X=x].

## Headline results (on synthetic data)
| Metric | Value |
|---|---|
| Qini lift over random (T-Learner) | see dashboard |
| ROI gain vs random targeting @ 30% budget | see dashboard |
| Placebo ATE (shuffled T) | ≈ 0 (confirms no spurious signal) |
| Recovered ATE vs True ATE | within ±5% |

## Why synthetic data?
With real observational data, we never observe both Y(0) and Y(1) for the same person (the fundamental problem of causal inference). Synthetic data gives us the **ground-truth τ(x) for every unit**, so we can measure:
- How closely each meta-learner recovers the true CATE
- Whether ranking by predicted uplift actually captures the highest-uplift units
- Whether the placebo refutation correctly collapses the signal

This is a **methods showcase**: the algorithms transfer to any real domain; the synthetic data simply lets us validate them honestly.

## Methods
### Meta-learners (all implemented from scratch on scikit-learn)
- **S-Learner**: single model on [X, T]; τ̂(x) = f(x,1) − f(x,0)
- **T-Learner**: separate models on treated/control; τ̂(x) = f₁(x) − f₀(x)
- **X-Learner**: T-learner → impute individual effects → model them, weighted by propensity score
- **CausalForest (EconML)**: if `econml` is installed, also runs `CausalForestDML`

### Evaluation
- **Qini coefficient**: area between the model's uplift curve and a random-targeting baseline
- **True-tau correlation/MSE**: direct comparison against known ground truth (synthetic-data advantage)
- **Policy analysis**: compare model targeting vs random vs treat-all at a given budget
- **Placebo refutation**: shuffle T, refit, confirm ATE collapses → confirms real estimates aren't spurious

## Run instructions
```bash
pip install -r requirements.txt

python -m src.data        # smoke test: prints true ATE, saves histogram
python -m src.evaluate    # fits all learners, prints metrics, saves qini_plot.png
streamlit run app.py      # interactive dashboard
```

## Project structure
```
├── src/
│   ├── data.py        # synthetic data generator (known ground-truth CATE)
│   ├── learners.py    # S/T/X-learner + optional EconML wrapper
│   └── evaluate.py    # Qini curves, policy report, placebo test
├── app.py             # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Data generating process
- **N = 20,000** synthetic individuals with 5 covariates (age, income, recency, channel, loyal)
- **Confounding**: older/higher-income users are more likely treated AND have higher baseline outcomes
- **Overlap**: propensity clipped to [0.1, 0.9] — every covariate region has both treated/control
- **CATE heterogeneity**: loyal + young + recent → high uplift; some units have τ(x) < 0
- **Outcome**: Y = μ₀(x) + T·τ(x) + ε, continuous, no sigmoid (clean gradients for validation)
