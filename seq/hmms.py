import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import pymc as pymc


# Load data
df = pd.read_csv(pymc.get_data("deaths_and_temps_england_wales.csv"))

# Discretise temperature into 3 bins
df['temp_discrete'] = pd.qcut(df['temp'], q=3, labels=[0, 1, 2])

# Discretise deaths into 3 bins
df['deaths_discrete'] = pd.qcut(df['deaths'], q=3, labels=[0, 1, 2])

# State and observation sequences
states = df['temp_discrete'].astype(int).values
observations = df['deaths_discrete'].astype(int).values

def train_HMM1(states, observations):
    n_states = len(np.unique(states))
    n_obs = len(np.unique(observations))
    T = len(states)

    # Start probs
    startprob = np.zeros(n_states)
    startprob[states[0]] = 1.0

    # Transition matrix
    transmat = np.zeros((n_states, n_states))
    for i in range(T - 1):
        transmat[states[i], states[i+1]] += 1
    for i in range(n_states):
        row_sum = transmat[i].sum()
        if row_sum > 0:
            transmat[i] = transmat[i] / row_sum 
        else:
            np.ones(n_states)/n_states

    # Emission matrix
    emissionprob = np.zeros((n_states, n_obs))
    for i in range(T):
        emissionprob[states[i], observations[i]] += 1
    for i in range(n_states):
        row_sum = emissionprob[i].sum()
        if row_sum > 0:
            emissionprob[i] = emissionprob[i] / row_sum 
        else:
            np.ones(n_obs)/n_obs

    # Build HMM1
    model = hmm.CategoricalHMM(n_components=n_states)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob

    return model


def train_HMM2(observations):

    X = observations.reshape(-1, 1)
    n_states = len(np.unique(observations))

    model = hmm.CategoricalHMM(
        n_components=n_states,
        n_iter=300,
        random_state=42
    )
    model.fit(X)
    return model

def plot_sequences_and_distribution(observations, pred1, pred2):

    plt.figure(figsize=(12, 10))

    # Plot 1: Actual
    plt.subplot(3, 1, 1)
    plt.plot(observations, color='steelblue', linewidth=1.3)
    plt.title("Actual Death Categories")
    plt.xlabel("Time (months)")
    plt.ylabel("Death Category")
    plt.yticks([0, 1, 2])
    plt.grid(alpha=0)

    # Plot 2: HMM1
    plt.subplot(3, 1, 2)
    plt.plot(pred1, color='coral', linewidth=1.3)
    plt.title("HMM1 Predicted Death Categories (Supervised)")
    plt.xlabel("Time (months)")
    plt.ylabel("Death Category")
    plt.yticks([0, 1, 2])
    plt.grid(alpha=0)

    # Plot 3: HMM2
    plt.subplot(3, 1, 3)
    plt.plot(pred2, color='seagreen', linewidth=1.3)
    plt.title("HMM2 Predicted Death Categories (Unsupervised)")
    plt.xlabel("Time (months)")
    plt.ylabel("Death Category")
    plt.yticks([0, 1, 2])
    plt.grid(alpha=0)

    plt.tight_layout()
    plt.savefig("death_sequences2.png", dpi=300, bbox_inches='tight')
    plt.show()


    # Save distribution plot 
    actual_counts = np.bincount(observations, minlength=3)
    hmm1_counts = np.bincount(pred1, minlength=3)
    hmm2_counts = np.bincount(pred2, minlength=3)

    actual_prop = actual_counts / actual_counts.sum()
    hmm1_prop = hmm1_counts / hmm1_counts.sum()
    hmm2_prop = hmm2_counts / hmm2_counts.sum()

    categories = ["Low", "Medium", "High"]
    x = np.arange(3)
    width = 0.25

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width, actual_prop, width, label="Actual")
    bars2 = plt.bar(x,         hmm1_prop, width, label="HMM1")
    bars3 = plt.bar(x + width, hmm2_prop, width, label="HMM2")

    plt.xticks(x, categories)
    plt.ylabel("Proportion")
    plt.title("Death Category Distribution Comparison")
    plt.legend()

    plt.grid(axis='y', alpha=0)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.01,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig("death_distribution2.png", dpi=300, bbox_inches='tight')
    plt.show()


HMM1 = train_HMM1(states, observations)
HMM2 = train_HMM2(observations)

'''
_, pred1 = HMM1.sample(len(observations))
_, pred2 = HMM2.sample(len(observations))
'''
# We use predict() here instead of sample() because sample generates a new
# synthetic sequence from the model, which introduces randomness and makes
# KL/error comparisons unstable across runs. predict() gives the most likely
# hidden-state sequence for the actual observations, producing deterministic
# outputs that allow fair, repeatable evaluation of HMM1 vs HMM2.

pred1 = HMM1.predict(observations.reshape(-1, 1))
pred2 = HMM2.predict(observations.reshape(-1, 1))


pred1 = pred1.flatten()
pred2 = pred2.flatten()

plot_sequences_and_distribution(observations, pred1, pred2)


# KL Divergence
def kl(p, q):
    return np.sum(p * np.log((p + 1e-12) / (q + 1e-12)))


hmm1_kl = kl(observations, pred1)
hmm2_kl = kl(observations, pred2)
print(f"HMM1 KL: {hmm1_kl}")
print(f"HMM2 KL: {hmm2_kl}")


# distribution Error
hmm1_dist = np.bincount(pred1, minlength=3) / len(pred1)
hmm2_dist = np.bincount(pred2, minlength=3) / len(pred2)
actual_dist = np.bincount(observations, minlength=3) / len(observations)

hmm1_error = np.sum(np.abs(hmm1_dist - actual_dist))
hmm2_error = np.sum(np.abs(hmm2_dist - actual_dist))
print("HMM 1 distribution error: ", hmm1_error)
print("HMM 2 absolute error:", hmm2_error )


# Log-likelihoods
loglik1 = HMM1.score(observations.reshape(-1, 1))
loglik2 = HMM2.score(observations.reshape(-1, 1))

print("HMM1 log-likelihood:", loglik1)
print("HMM2 log-likelihood:", loglik2)
