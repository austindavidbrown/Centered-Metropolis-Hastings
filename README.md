# Centered Metropolis-Hastings independence algorithm for Bayesian logistic regression with Gaussian priors

A Python implementation of centered Metropolis-Hastings for Bayesian logistic regression with Gaussian priors for https://arxiv.org/abs/2111.10406 to appear in Journal of Applied Probability. This library uses Pytorch for matrix calculations. Install using PIP:

```bash
pip install cmhi
```

# Example

Here is a simple example:

```python
import torch
from cmhi import BayesianLogisticRegression

# Generate data
n_features = 10
n_samples = 200
bias_true = 1
theta_true = torch.zeros(n_features).uniform_(-1, 1)
X = n_samples**(-1/2) * torch.zeros(n_samples, n_features).uniform_(-1, 1)
Y = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(bias_true + X @ theta_true)
for i in range(0, Y.size(0)):
  Y[i] = torch.bernoulli(prob[i])

# Centered Metropolis-Hastings independence sampler
bayesian_logistic_regression = BayesianLogisticRegression(X, Y, Cov_prior = 10 * torch.eye(n_features))
bias_mle, thetas, accepts = bayesian_logistic_regression.sample(n_iterations = 10**4)

print("The MLE is used for the bias:", bias_mle)
print("Number of accepted samples from the proposal:", accepts.sum().item())
```

## Citation

```
@article{brown:jones:2024,
    title={{Exact Convergence Analysis for Metropolis-Hastings Independence Samplers in Wasserstein Distances}}, 
    author={Austin Brown and Galin Jones},
    year={2024},
    journal = {To appear Journal of Applied Probability 61.1}
}
```

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)
