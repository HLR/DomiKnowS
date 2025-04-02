## Proposed interface for non-categorical values in DomiKnowS
See `proposed_graph.py`.

## Running mnist-arithmetic with integer approximations
`python run.py --device {cpu, cuda}` runs all four methods for approximating mnist-sum with integer values. Metrics across all operations/epochs are saved to `results/` by default (see: `training.py`).

- `ste_separate_gumbel`: Straight-through estimator of only the one-hot operation w/ Gumbel noise (see below).
- `ste_separate`: Straight-through estimator of only the one-hot operation w/o Gumbel noise
- `ste_full`: Straight-through estimator of the full argmax operation (see below).
- `weighted_sum`: Performs a weighted sum of the digit values (0, ..., 9), weigted wrt the predicted probabilities (softmax of the logits).

All implementations can be found in the `approximations` dictionary in `run.py`.

## Straight-through estimator method
Let $p_1, p_2 \in \mathbb{R}^{10}$ be the probabilities of the NN for digit 1 and digit 2, respectively, calculated with a softmax over the logits.

We have some discrete transformation $f: \mathbb{R}^{10} \rightarrow \{0, ..., 9\}$ that returns the argmax of the input logits.

During the forward pass we use $f$ as normal, however, during the backward pass, we want to approximate the gradients for $f$.

One option is: approximate the gradients of $f$ by calculating gradients without $f$

In other words, when calculating e.g., $\frac{d f(p_i)}{d \theta}$ for model parameters $\theta$, we just use $\frac{d p_i}{d \theta}$.

Another option is: separate out $f$ into a continuous step and a discrete step, and do the approximation on only the discrete step

The argmax operation $f$ can be expressed as $f(p_i) = \text{OneHot}(p_i) \cdot [0, ..., 9]$, where $\text{OneHot}: \mathbb{R}^{10} \rightarrow \{0, 1\}^{10}$ maps a continuous vector to a one-hot vector for the maximum value.

We then only need to perform an approximation for $\text{OneHot}(p_i)$ (using the same approach as before).

The latter approach follows this paper: https://arxiv.org/pdf/2109.08512

To completely follow the approach from that paper, we can add Gumbel noise to the logits before we perform Softmax, making it a more faithful approximation of the target distribution.
