# Centered Metropolis-Hastings independence algorithm for Bayesian logistic regression with Gaussian priors

A Python implementation of centered Metropolis-Hastings for Bayesian logistic regression with Gaussian priors from the pre-print https://arxiv.org/abs/2111.10406. This library uses Pytorch for matrix calculations. Install using PIP:

```bash
pip install cmhi
```

# Example

Here is a simple example:

```python

import torch
from cmhi import BayesianLogisticRegression

n_features = 100
n_samples = 10

# Generate data
bias_true = 1
theta_true = torch.zeros(n_features).normal_(0, 1)
X = torch.zeros(n_samples, n_features).uniform_(-1, 1)
Y = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(bias_true + X @ theta_true)
for i in range(0, n_samples):
  Y[i] = torch.bernoulli(prob[i])


# Centered Metropolis-Hastings independence sampler with Gaussian proposal
# The covariance for the proposal can be tuned
bayesian_logistic_regression = BayesianLogisticRegression(X, Y,  
                                                          Cov_prior = torch.eye(n_features),
                                                          Cov_proposal = .9 * torch.eye(n_features))
bias_mle, thetas, accepts = bayesian_logistic_regression.sample(n_iterations = 10**4)

print("The MLE is used for the bias:", bias_mle)
print("Number of accepted samples from the proposal:", int(accepts.sum().item()))
```

## Citation

https://arxiv.org/abs/2111.10406

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)
