"""
Synthetic data generator for uplift modeling.

Ground-truth CATE tau(x) is known, enabling crisp validation that is impossible
with real observational data. That is the entire point of synthetic data here.
"""
import numpy as np
import pandas as pd


def generate_data(n: int = 20_000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, float]:
    """
    Returns (df, true_tau, true_ate).

    Covariates
    ----------
    age         continuous, roughly 18-70
    income      continuous, log-normal-ish
    recency     continuous, days since last purchase
    channel     binary, online vs offline
    loyal       binary, loyalty member flag

    Confounding: high-age / high-income users are more likely to be treated AND
    have higher baseline outcomes, so naive (treated - control) is biased.

    Treatment effect heterogeneity
    --------------------------------
    tau(x) = 3 * loyal + 2 * (age_scaled) - 1.5 * (recency_scaled) + eps_tau
    => loyal, younger, recent users respond most; some users have slightly negative tau
    """
    rng = np.random.default_rng(seed)

    age = rng.uniform(18, 70, n)
    income = rng.lognormal(mean=10.5, sigma=0.6, size=n)
    recency = rng.exponential(scale=30, size=n)
    channel = rng.binomial(1, 0.55, n).astype(float)
    loyal = rng.binomial(1, 0.3, n).astype(float)

    age_s = (age - age.mean()) / age.std()
    income_s = (np.log(income) - np.log(income).mean()) / np.log(income).std()
    recency_s = (recency - recency.mean()) / recency.std()

    # propensity: older, higher-income, online users more likely to be treated
    log_odds = 0.4 * age_s + 0.5 * income_s - 0.3 * recency_s + 0.2 * channel
    propensity = 1 / (1 + np.exp(-log_odds))
    # clip to ensure overlap
    propensity = np.clip(propensity, 0.1, 0.9)

    treatment = rng.binomial(1, propensity).astype(float)

    # baseline outcome: age, income, loyal all drive it positively
    mu0 = (
        2.0 * age_s
        + 1.5 * income_s
        + 0.5 * loyal
        - 0.4 * recency_s
        + 10.0
    )

    # true CATE: loyal + young + recent → high uplift; some slightly negative
    true_tau = (
        3.0 * loyal
        + 2.0 * (1 - (age - 18) / 52)   # younger → higher
        - 1.5 * recency_s                # recent → higher
        - 0.5                            # shift so some are negative
    )

    noise = rng.normal(0, 1.5, n)
    outcome = mu0 + treatment * true_tau + noise

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "recency": recency,
        "channel": channel,
        "loyal": loyal,
        "treatment": treatment,
        "outcome": outcome,
        "propensity": propensity,
    })

    true_ate = float(true_tau.mean())
    return df, true_tau, true_ate


FEATURE_COLS = ["age", "income", "recency", "channel", "loyal"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df, true_tau, true_ate = generate_data()
    print(f"Rows: {len(df)}")
    print(f"Treatment rate: {df['treatment'].mean():.3f}")
    print(f"True ATE: {true_ate:.4f}")
    print(f"tau(x) range: [{true_tau.min():.2f}, {true_tau.max():.2f}]")
    print(f"Outcome mean: {df['outcome'].mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(true_tau, bins=60, color="steelblue", edgecolor="white")
    axes[0].axvline(true_ate, color="red", linestyle="--", label=f"ATE = {true_ate:.2f}")
    axes[0].set_title("True CATE distribution")
    axes[0].set_xlabel("tau(x)")
    axes[0].legend()

    axes[1].hist(df["propensity"], bins=40, color="orange", edgecolor="white")
    axes[1].set_title("Propensity score distribution (overlap check)")
    axes[1].set_xlabel("P(T=1 | X)")

    plt.tight_layout()
    plt.savefig("true_tau_hist.png", dpi=120)
    print("Saved true_tau_hist.png")
    plt.show()
