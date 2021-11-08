import torch

class BayesianLogisticRegression:
  # Gradient descent with annealing step sizes
  @staticmethod
  def graddescent(X, Y, sigma2_prior, C, 
                  stepsize = .1, tol = 10**(-10), max_iterations = 10**5):
    C_half = torch.linalg.cholesky(C)
    C_inv = torch.cholesky_inverse(C_half)
    bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    b = torch.zeros(1)
    theta = torch.zeros(X.size(1))

    old_loss = bceloss(b + X @ theta, Y.double()) \
               + 1/(2.0 * sigma2_prior) * theta @ C_inv @ theta

    for t in range(1, max_iterations):
      grad_loss_b = torch.ones(X.size(0)) @ (torch.sigmoid(b + X @ theta) - Y)
      grad_loss_theta = X.T @ (torch.sigmoid(b + X @ theta) - Y) + 1/(sigma2_prior) * C_inv @ theta

      if torch.any(torch.isnan(grad_loss_b)) or torch.any(torch.isnan(grad_loss_theta)):
        raise Exception("NAN value in gradient descent.")
      else:
        b_new = b - stepsize * grad_loss_b
        theta_new = theta - stepsize * grad_loss_theta
        new_loss = bceloss(b_new + X @ theta_new, Y.double()) \
                   + 1/(2.0 * sigma2_prior) * theta_new @ C_inv @ theta_new
        
        # New loss worse than old loss? Reduce step size and try again.
        if (new_loss > old_loss):
          stepsize = stepsize * (.99)
        else:
          # Stopping criterion
          if (old_loss - new_loss) < tol:
            return b, theta

          # Update
          b = b_new
          theta = theta_new
          old_loss = new_loss

    raise Exception("Gradient descent failed to converge.")


  # MHI sampler using 'centered' proposal for Bayesian logistic regression
  def __init__(self, X, Y, sigma2_prior, 
               C, h = 1,
               stepsize_opt = .1, tol_opt = 10**(-10), max_iterations_opt = 10**5):
    self.dimension = X.size(1)
    self.X = X
    self.Y = Y
    self.sigma2_prior = sigma2_prior

    C_half = torch.linalg.cholesky(C)
    C_inv = torch.cholesky_inverse(C_half)

    self.C_half = C_half
    self.C_inv = C_inv
    self.h = h

    # Optimize target
    b_opt, theta_opt = BayesianLogisticRegression.graddescent(X, Y, sigma2_prior, C,
                                   stepsize_opt, tol_opt, max_iterations_opt)
    self.b_opt = b_opt
    self.theta_opt = theta_opt

  def sample(self, theta_0 = None, n_iterations = 1):
    accepts = torch.zeros(n_iterations)
    bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    if theta_0 is None:
      theta_0 = self.theta_opt
      f_proposal_theta = torch.zeros(1)
    else:
      f_proposal_theta = 1/(2.0 * self.h) * (theta_0 - self.theta_opt) @ self.C_inv @ (theta_0 - self.theta_opt)

    # Compute the previous theta using the opt
    f_target_theta = bceloss(self.b_opt + self.X @ theta_0, self.Y.double()) \
                     + 1/(2.0 * self.sigma2_prior) * theta_0 @ self.C_inv @ theta_0

    thetas = torch.zeros(n_iterations, self.dimension)
    thetas[0] = theta_0
    for t in range(1, n_iterations):
      xi = torch.zeros(self.theta_opt.size(0)).normal_(0, 1)
      theta_new = self.theta_opt + self.h**(1/2) * self.C_half @ xi

      # MH step
      f_proposal_theta_new = 1/(2.0) * xi.pow(2).sum()
      f_target_theta_new = bceloss(self.b_opt + self.X @ theta_new, self.Y.double()) \
                           + 1/(2.0 * self.sigma2_prior) * theta_new @ self.C_inv @ theta_new
      u_sample = torch.zeros(1).uniform_(0, 1)
      if torch.log(u_sample) <= f_proposal_theta_new - f_target_theta_new + f_target_theta - f_proposal_theta:  
        thetas[t] = theta_new

        # Update the previous iteration values if accepted
        f_proposal_theta = f_proposal_theta_new
        f_target_theta = f_target_theta_new

        accepts[t] = 1
      else:
        thetas[t] = thetas[t-1]

    return self.b_opt, thetas, accepts

  def sample_until_accept(self, theta_0 = None, max_iterations = 10**(5)):
    bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    if theta_0 is None:
      theta_0 = self.theta_opt
      f_proposal_theta = torch.zeros(1)
    else:
      f_proposal_theta = 1/(2.0 * self.h) * (theta_0 - self.theta_opt) @ self.C_inv @ (theta_0 - self.theta_opt)

    # Compute the previous theta using the opt
    f_target_theta = bceloss(self.b_opt + self.X @ theta_0, self.Y.double()) \
                     + 1/(2.0 * self.sigma2_prior) * theta_0 @ self.C_inv @ theta_0

    for t in range(1, max_iterations):
      xi = torch.zeros(self.theta_opt.size(0)).normal_(0, 1)
      theta_new = self.theta_opt + self.h**(1/2) * self.C_half @ xi

      # MH step
      f_proposal_theta_new = 1/(2.0) * xi.pow(2).sum()
      f_target_theta_new = bceloss(self.b_opt + self.X @ theta_new, self.Y.double()) \
                           + 1/(2.0 * self.sigma2_prior) * theta_new @ self.C_inv @ theta_new
      u_sample = torch.zeros(1).uniform_(0, 1)
      if torch.log(u_sample) <= f_proposal_theta_new - f_target_theta_new + f_target_theta - f_proposal_theta:  
        return self.b_opt, theta_new

    raise Exception("Failed to accept from the proposal.")



'''
# Test example

import torch
from cmhi import BayesianLogisticRegression

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


# CMHI Sampler
bayesian_logistic_regression = BayesianLogisticRegression(X, Y, sigma2_prior,  
                                                          C = torch.eye(n_features),
                                                          h = .9 * sigma2_prior)
bias, thetas, accepts = bayesian_logistic_regression.sample(n_iterations = 10**4)

print("CMHI Sampler:")
print("n accepts:", int(accepts.sum().item()))

predictions = torch.round(torch.sigmoid(bias + X @ thetas.mean(0))).long()
accuracy = 1/Y.size(0)*torch.sum(predictions == Y).item()
print("accuracy:", accuracy)
'''