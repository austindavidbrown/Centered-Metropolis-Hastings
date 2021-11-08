# 'Centered' Metropolis-Hastings Independence algorithm for Bayesian logistic regression

A Python library for 'centered' Metropolis-Hastings using Pytorch for tensor computations. Install using PIP:

```bash
pip install cmhi
```

# Example

Here is a simple example:

```python
import torch
from cmhi import bayesian_logistic_regression

n_features = 100
n_samples = 10
sigma2_prior = 1

# Generate data
b_true = 1
theta_true = torch.zeros(n_features).normal_(0, sigma2_prior**(1/2))
X = torch.zeros(n_samples, n_features).uniform_(-1, 1)
Y = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(b_true + X @ theta_true)
for i in range(0, Y.size(0)):
  Y[i] = torch.bernoulli(prob[i])
  
# Sample
bias, thetas, accepts = bayesian_logistic_regression(X, Y, sigma2_prior = sigma2_prior, C = torch.eye(n_features),
                                                     n_iterations = 10**4, h = .9 * sigma2_prior)
  

predictions = torch.round(torch.sigmoid(bias + X @ thetas.mean(0))).long()
accuracy = 1/Y.size(0)*torch.sum(predictions == Y)

print("n accepts:", int(accepts.sum().item()))
print("accuracy:", accuracy.item())
```

## Citation

## Authors

Austin Brown (graduate student at the School of Statistics, University of Minnesota)

## Dependencies

* [Python](https://www.python.org)
* [PyTorch](http://pytorch.org/)
