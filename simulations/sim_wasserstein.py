import torch

from cmhi import BayesianLogisticRegression

# Upper bound estimate of the Wasserstein distance empirically using the synchronous coupling
def estimate_wasserstein(X, Y, sigma2_prior, C,
                         b_opt, theta_opt, theta_target, 
                         n_iterations, h):
  bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")
  C_half = torch.linalg.cholesky(C)
  C_inv = torch.cholesky_inverse(C_half)

  thetas = torch.zeros(n_iterations, X.size(1))
  thetas_target = torch.zeros(n_iterations, X.size(1))

  f_theta = bceloss(b_opt + X @ theta_opt, Y.double()) \
            + 1/(2.0 * sigma2_prior) * theta_opt @ C_inv @ theta_opt
  f_proposal_theta = torch.zeros(1)
  
  f_theta_target = bceloss(b_opt + X @ theta_target, Y.double()) \
                   + 1/(2.0 * sigma2_prior) * theta_target @ C_inv @ theta_target
  f_proposal_theta_target =  1/(2.0 * h) * (theta_target - theta_opt) @ C_inv @ (theta_target - theta_opt)
  
  thetas[0] = theta_opt
  thetas_target[0] = theta_target
  for t in range(1, n_iterations):
    xi = torch.zeros(theta_opt.size(0)).normal_(0, 1)
    theta_new = theta_opt + h**(1/2) * C_half @ xi

    # MH step
    u_sample = torch.zeros(1).uniform_(0, 1)
    f_proposal_theta_new = 1/(2.0) * xi.pow(2).sum()
    f_theta_new = bceloss(b_opt + X @ theta_new, Y.double()) \
                  + 1/(2.0 * sigma2_prior) * theta_new @ C_inv @ theta_new

    both_accepted = True
    if torch.log(u_sample) <= f_theta - f_theta_new + f_proposal_theta_new - f_proposal_theta:  
      thetas[t] = theta_new

      # Update the previous iteration values if accepted
      f_proposal_theta = f_proposal_theta_new
      f_theta = f_theta_new
    else:
      both_accepted = False
      thetas[t] = thetas[t-1]

    if torch.log(u_sample) <= f_theta_target - f_theta_new + f_proposal_theta_new - f_proposal_theta_target:  
      thetas_target[t] = theta_new

      # Update the previous iteration values if accepted
      f_proposal_theta_target = f_proposal_theta_new
      f_theta_target = f_theta_new
    else:
      both_accepted = False
      thetas_target[t] = thetas_target[t-1]

    # If both have accepted, the chains have coalesced  
    if both_accepted:
      return 0.0

  return torch.norm(thetas[-1] - thetas_target[-1]).item()




def run_sim(sigma2_prior):
  wasserstein_estimates = torch.zeros(n_simulations)
  std_wasserstein_estimates = torch.zeros(n_simulations)

  for sim in range(0, n_simulations):
    n_features = dimensions_list[sim]
    n_samples = samples_list[sim]
    sigma2_model = 1/n_samples

    # Generate data
    print("SIMULATION: n, d =", (n_samples, n_features))
    b_true = 1
    theta_true = torch.zeros(n_features).uniform_(-2, 2)
    X = torch.zeros(n_samples, n_features).normal_(0, sigma2_model**(1/2.0))
    Y = torch.zeros(n_samples, dtype=torch.long)
    prob = torch.sigmoid(b_true + X @ theta_true)
    for i in range(0, Y.size(0)):
      Y[i] = torch.bernoulli(prob[i])

    C = 1/n_features * torch.eye(n_features)
    h = .8 * sigma2_prior
    bayesian_logistic_regression = BayesianLogisticRegression(X, Y, sigma2_prior, 
                                                              C, h)
    unbiased_wasserstein_estimates = torch.zeros(n_iid)
    for j in range(0, n_iid):
      print("Estimating Wasserstein", j + 1, "of", n_iid)
      # Generate a sample using CMHI sampler
      b_opt, theta_target = bayesian_logistic_regression.sample_until_accept()

      unbiased_wasserstein_estimates[j] = estimate_wasserstein(X, Y, sigma2_prior, C,
                                                               b_opt = bayesian_logistic_regression.b_opt, 
                                                               theta_opt = bayesian_logistic_regression.theta_opt, 
                                                               theta_target = theta_target, 
                                                               n_iterations = n_iterations, 
                                                               h = h)

    wasserstein_estimates[sim] = unbiased_wasserstein_estimates.mean(0)
    std_wasserstein_estimates[sim] = unbiased_wasserstein_estimates.std(0)
    print("Wasserstein estimate:", wasserstein_estimates[sim].item())

  return wasserstein_estimates, std_wasserstein_estimates



###
# Simulations
###
torch.manual_seed(5)
torch.set_default_dtype(torch.float32)
torch.autograd.set_grad_enabled(False) 

n_iid = 1000
n_iterations = 10**(4)

dimensions_list = [100, 200, 500, 1000]
samples_list = [int(1/10 * d) for d in dimensions_list]
n_simulations = len(dimensions_list)


# Smaller prior variance
sigma2_prior = 10**(1)
print("Simulating for sigma2_prior = ", sigma2_prior)
print("--------------------------------")
wasserstein_estimates, std_wasserstein_estimates = run_sim(sigma2_prior)
torch.save((wasserstein_estimates, std_wasserstein_estimates), "sim_wasserstein1.pt")

# Larger prior variance
sigma2_prior = 10**(4)
print("Simulating for sigma2_prior = ", sigma2_prior)
print("--------------------------------")
mean_wasserstein_estimates2, std_wasserstein_estimates2 = run_sim(sigma2_prior)
torch.save((mean_wasserstein_estimates2, std_wasserstein_estimates2), "sim_wasserstein2.pt")


###
# Plot
###
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

mean_wasserstein_estimates1, std_wasserstein_estimates1 = torch.load("sim_wasserstein1.pt")
mean_wasserstein_estimates2, std_wasserstein_estimates2 = torch.load("sim_wasserstein2.pt")


iterations = torch.arange(0, n_simulations)
samples_and_dimensions = list(zip(dimensions_list, samples_list))

linewidth = 3
markersize = 8
alpha = .8

light_blue_color = (3./255, 37./255, 76./255)
dark_blue_color = (24./255, 123./255, 205./255)


plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

plt.plot(iterations, mean_wasserstein_estimates1.cpu().numpy(), 
         '-', alpha = alpha, marker="v", markersize=markersize, color=dark_blue_color, label=r"Smaller prior variance", linewidth = linewidth)
plt.fill_between(iterations, mean_wasserstein_estimates1 - std_wasserstein_estimates1,
                 mean_wasserstein_estimates1 + std_wasserstein_estimates1, alpha=0.1,
                 color=dark_blue_color)

plt.plot(iterations, mean_wasserstein_estimates2.cpu().numpy(), 
         '-', alpha = alpha, marker="v", markersize=markersize, color=light_blue_color, label=r"Larger prior variance", linewidth = linewidth)
plt.fill_between(iterations, mean_wasserstein_estimates2 - std_wasserstein_estimates2,
                 mean_wasserstein_estimates2 + std_wasserstein_estimates2, alpha=0.1,
                 color=light_blue_color)

plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Estimated Wasserstein distance", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("wasserstein_plot.png", pad_inches=0, bbox_inches='tight',)

