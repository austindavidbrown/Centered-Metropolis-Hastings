import torch

class BayesianLogisticRegression:
  # Gradient descent with annealing step sizes
  @staticmethod
  def graddescent(X, Y, Cov_prior, 
                  stepsize = .1, tol = 10**(-10), max_iterations = 10**5):
    Cov_prior_inv = torch.cholesky_inverse(torch.linalg.cholesky(Cov_prior))
    bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    b = torch.zeros(1)
    theta = torch.zeros(X.size(1))

    old_loss = bceloss(b + X @ theta, Y.double()) \
               + 1/(2.0) * theta @ Cov_prior_inv @ theta

    for t in range(1, max_iterations):
      grad_loss_b = torch.ones(X.size(0)) @ (torch.sigmoid(b + X @ theta) - Y)
      grad_loss_theta = X.T @ (torch.sigmoid(b + X @ theta) - Y) + Cov_prior_inv @ theta

      if torch.any(torch.isnan(grad_loss_b)) or torch.any(torch.isnan(grad_loss_theta)):
        raise Exception("NAN value in gradient descent.")
      else:
        b_new = b - stepsize * grad_loss_b
        theta_new = theta - stepsize * grad_loss_theta
        new_loss = bceloss(b_new + X @ theta_new, Y.double()) \
                   + 1/(2.0) * theta_new @ Cov_prior_inv @ theta_new
        
        # New loss worse than old loss? Reduce step size and try again.
        if (new_loss > old_loss):
          stepsize = stepsize * (.99)
        else:
          # Stopping criterion
          if (old_loss - new_loss) < tol:
            return b_new, theta_new

          # Update
          b = b_new
          theta = theta_new
          old_loss = new_loss

    raise Exception("Gradient descent failed to converge.")


  # MHI sampler using 'centered' proposal for Bayesian logistic regression
  # X : matrix of features
  # Y : vector of responses
  # Cov_prior : prior covariance matrix
  # stepsize_opt : initial step size for gradient descent
  # tol_opt : gradient descent tolerance
  # max_iterations_opt : maximum iterations for gradient descent
  def __init__(self, X, Y, Cov_prior, 
               stepsize_opt = .1, tol_opt = 10**(-10), max_iterations_opt = 10**5):
    self.dimension = X.size(1)
    self.X = X
    self.Y = Y

    self.Cov_prior = Cov_prior
    self.Cov_prior_inv = torch.cholesky_inverse(torch.linalg.cholesky(self.Cov_prior))

    self.Cov_proposal = Cov_prior
    self.Cov_proposal_L = torch.linalg.cholesky(self.Cov_proposal)
    self.Cov_proposal_inv = torch.cholesky_inverse(self.Cov_proposal_L)

    # Optimize target
    b_opt, theta_opt = BayesianLogisticRegression.graddescent(X, Y, Cov_prior,
                                                              stepsize_opt, tol_opt, max_iterations_opt)
    self.b_opt = b_opt
    self.theta_opt = theta_opt
    self.grad_f_theta_opt = self.X.T @ (torch.sigmoid(self.b_opt + self.X @ self.theta_opt) - self.Y) \
                            + self.Cov_prior_inv @ self.theta_opt

  # Generate samples
  # The bias or intercept term is not sampled but instead the MLE is used instead
  # theta_0 : initial starting point
  # n_iterations : number of iterations
  def sample(self, theta_0 = None, n_iterations = 1):
    accepts = torch.zeros(n_iterations)
    bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")

    if theta_0 is None:
      theta_0 = self.theta_opt

    f_proposal_theta = 1/(2.0) * (theta_0 - self.theta_opt + self.Cov_proposal @ self.grad_f_theta_opt) \
                                  @ self.Cov_proposal_inv @ (theta_0 - self.theta_opt + self.Cov_proposal @ self.grad_f_theta_opt)
    f_target_theta = bceloss(self.b_opt + self.X @ theta_0, self.Y.double()) \
                     + 1/(2.0) * theta_0 @ self.Cov_prior_inv @ theta_0

    thetas = torch.zeros(n_iterations, self.dimension)
    thetas[0] = theta_0
    for t in range(1, n_iterations):
      xi = torch.zeros(self.dimension).normal_(0, 1)
      theta_new = self.theta_opt - self.Cov_proposal @ self.grad_f_theta_opt \
                  + self.Cov_proposal_L @ xi

      # MH step
      f_proposal_theta_new = 1/(2.0) * xi.pow(2).sum()
      f_target_theta_new = bceloss(self.b_opt + self.X @ theta_new, self.Y.double()) \
                           + 1/(2.0) * theta_new @ self.Cov_prior_inv @ theta_new
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

  
 



'''
# Test example
torch.manual_seed(0)

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


X_new = torch.zeros(n_samples, n_features)
for i in range(0, n_samples):
  X_new[i, :] = 1/(n_features) * torch.zeros(n_features).uniform_(-1, 1)
Y_new = torch.zeros(n_samples, dtype=torch.long)
prob = torch.sigmoid(bias_true + X_new @ theta_true)
for i in range(0, Y_new.size(0)):
  Y_new[i] = torch.bernoulli(prob[i])

predictions = torch.round(torch.sigmoid(bias_mle + X_new @ thetas.mean(0))).long()
accuracy = 1/Y_new.size(0)*torch.sum(predictions == Y_new).item()
print("accuracy:", accuracy)

predictions_max = torch.round(torch.sigmoid(bias_mle + X_new @ thetas[0])).long()
accuracy_max = 1/Y_new.size(0)*torch.sum(predictions_max == Y_new).item()
print("max accuracy:", accuracy_max)
'''
