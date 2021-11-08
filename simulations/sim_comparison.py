import time
import torch

import numpy as np
from pypolyagamma import BernoulliRegression

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns

def bayesian_logistic_regression_with_iteration_times(X, Y, sigma2_prior, C,
                                                      n_iterations, h,
                                                      stepsize_opt = .1, tol_opt = 10**(-10), max_iterations_opt = 10**5):
  iteration_times = torch.zeros(n_iterations)
  accepts = torch.zeros(n_iterations)
  bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")
  C_half = torch.linalg.cholesky(C)
  C_inv = torch.cholesky_inverse(C_half)

  # Optimize target
  b_opt, theta_opt = graddescent(X, Y, sigma2_prior, C,
                                 stepsize_opt, tol_opt, max_iterations_opt)
  # Compute the previous theta using the opt
  f_target_theta = bceloss(b_opt + X @ theta_opt, Y.double()) \
                   + 1/(2.0 * sigma2_prior) * theta_opt @ C_inv @ theta_opt
  f_proposal_theta = torch.zeros(1)

  thetas = torch.zeros(n_iterations, X.size(1))
  thetas[0] = theta_opt
  for t in range(1, n_iterations):
    # Record iteration time
    timestart = time.time()

    xi = torch.zeros(theta_opt.size(0)).normal_(0, 1)
    theta_new = theta_opt + h**(1/2) * C_half @ xi

    # MH step
    f_proposal_theta_new = 1/(2.0) * xi.pow(2).sum()
    f_target_theta_new = bceloss(b_opt + X @ theta_new, Y.double()) \
                         + 1/(2.0 * sigma2_prior) * theta_new @ C_inv @ theta_new
    u_sample = torch.zeros(1).uniform_(0, 1)
    if torch.log(u_sample) <= f_proposal_theta_new - f_target_theta_new + f_target_theta - f_proposal_theta:  
      thetas[t] = theta_new

      # Update the previous iteration values if accepted
      f_proposal_theta = f_proposal_theta_new
      f_target_theta = f_target_theta_new

      accepts[t] = 1
    else:
      thetas[t] = thetas[t-1]
    
    # Record iteration time
    timeend = time.time()
    iteration_times[t] = timeend - timestart

  return b_opt, thetas, accepts, iteration_times


###
# HMC sampler
###
def hmc_bayesian_logistic_regression(bias, X, Y, sigma2_prior,
                                     n_iterations, h, sigma2_hmc = 10, n_leapfrogs = 10):
  accepts = torch.zeros(n_iterations)
  iteration_times = torch.zeros(n_iterations)

  bceloss = torch.nn.BCEWithLogitsLoss(reduction="sum")
  n_dimension = X.size(1)
  thetas = torch.zeros(n_iterations, n_dimension)
  ps = torch.zeros(n_iterations, n_dimension)

  # Compute the previous theta using the opt
  theta_0 = thetas[0]
  f_theta = bceloss(bias + X @ theta_0, Y.double()) \
            + 1/(2.0 * sigma2_prior) * theta_0.pow(2).sum()
  
  for t in range(1, n_iterations):
    # Record iteration time
    t0 = time.time()

    # Run the leapfrog numerical integrator
    p_0 = torch.zeros(n_dimension).normal_(0, 1)
    p_new = p_0
    theta_new = thetas[t - 1]
    for l in range(0, n_leapfrogs):  
      # p step
      grad_f_theta = X.T @ (torch.sigmoid(bias + X @ theta_new) - Y) \
                     + 1/(sigma2_prior) * theta_new
      p_new = p_new - h/2 * grad_f_theta

      # theta step
      theta_new = theta_new + h/sigma2_hmc * p_new

      # p step
      grad_f_theta = X.T @ (torch.sigmoid(bias + X @ theta_new) - Y) \
                     + 1/(sigma2_prior) * theta_new
      p_new = p_new - h/2 * grad_f_theta

    # MH Step
    f_theta_new = bceloss(bias + X @ theta_new, Y.double()) \
                  + 1/(2.0 * sigma2_prior) * theta_new.pow(2).sum()

    H_theta = f_theta + 1/(2.0 * sigma2_hmc) * p_0.pow(2).sum()              
    H_theta_new = f_theta_new + 1/(2.0 * sigma2_hmc) * p_new.pow(2).sum()
    u_sample = torch.zeros(1).uniform_(0, 1)
    if torch.log(u_sample) <= H_theta - H_theta_new:  
      thetas[t] = theta_new

      # Update the previous iteration values if accepted
      f_theta = f_theta_new

      accepts[t] = 1
    else:
      thetas[t] = thetas[t-1]

    # Record iteration time
    t1 = time.time()
    iteration_times[t] = t1 - t0

  return thetas, accepts, iteration_times



###
# Simulations
###
torch.manual_seed(5)
torch.set_default_dtype(torch.float64)
torch.autograd.set_grad_enabled(False)

n_iterations = 10**(4)

dimensions_list = [100, 200, 500, 1000]
samples_list = [int(1/10 *d) for d in dimensions_list]
n_reps = len(dimensions_list)

avg_iteration_times = torch.zeros(n_reps)
avg_iteration_times_pg = torch.zeros(n_reps)
avg_iteration_times_hmc = torch.zeros(n_reps)

mses = torch.zeros(n_reps)
mses_pg = torch.zeros(n_reps)
mses_hmc = torch.zeros(n_reps)



for rep in range(0, n_reps):
  n_features = dimensions_list[rep]
  n_samples = samples_list[rep]
  sigma2_model = 1/n_samples
  sigma2_prior = .1

  # slightly smaller than the theory says
  h_cmhi = .9 * sigma2_prior
  C = torch.eye(n_features)

  # Total integration length for hmc
  # Note: Cherry picked for simulation because it is too sensitive.
  sigma2_hmc = 1
  h_hmc = .001
  n_leapfrogs = 100 # int(1/(h_hmc))

  # Generate data
  print("SIMULATION: n, d =", (n_samples, n_features))
  b_true = 1
  theta_true = torch.zeros(n_features).uniform_(-1, 1)
  X = torch.zeros(n_samples, n_features).normal_(0, sigma2_model**(1/2.0))
  Y = torch.zeros(n_samples, dtype=torch.long)
  prob = torch.sigmoid(b_true + X @ theta_true)
  for i in range(0, Y.size(0)):
    Y[i] = torch.bernoulli(prob[i])

  ######
  # CMHI Sampler
  ######
  print("Running CMHI")
  b_opt, thetas, accepts, iteration_times = bayesian_logistic_regression_with_iteration_times(
                                              X = X, Y = Y, 
                                              sigma2_prior = sigma2_prior, C = C,
                                              n_iterations = n_iterations, h = h_cmhi)
  print("Done.")
  
  avg_iteration_times[rep] = iteration_times.mean()
  mses[rep] = (thetas - thetas.mean(0)).pow(2).mean()

  print("n accepts:", accepts.sum().item())
  print("MSE:", mses[rep].item())
  print("avg iteration time:", avg_iteration_times[rep].item())
  print("-----------------------")

  ######
  # HMC Sampler
  ######
  print("Running HMC")
  thetas_hmc, accepts_hmc, iteration_times_hmc = hmc_bayesian_logistic_regression(bias = b_opt,
                                                                                  X = X, Y = Y, sigma2_prior = sigma2_prior,
                                                                                  n_iterations = n_iterations, h = h_hmc, 
                                                                                  n_leapfrogs = n_leapfrogs, sigma2_hmc = sigma2_hmc)
  print("Done.")

  avg_iteration_times_hmc[rep] = iteration_times_hmc.mean()
  mses_hmc[rep] = (thetas_hmc - thetas_hmc.mean(0)).pow(2).mean()

  print("n accepts:", accepts_hmc.sum().item())
  print("MSE:", mses_hmc[rep].item())
  print("avg iteration time:", avg_iteration_times_hmc[rep].item())
  print("-----------------------")

  ######
  # PG Sampler
  ######
  print("Running Polya-Gamma")
  iteration_times_pg = torch.zeros(n_iterations)

  thetas_pg = np.zeros((n_iterations, n_features))
  bs_pg = np.zeros((n_iterations, 1))
  pg_reg = BernoulliRegression(1, n_features, 
                               sigmasq_b = sigma2_prior, sigmasq_A = sigma2_prior)
  
  for t in range(0, n_iterations):
    # Record iteration time
    t0 = time.time()

    pg_reg.resample((X.numpy(), Y.unsqueeze(-1).numpy()))
    bs_pg[t, :] = pg_reg.b
    thetas_pg[t, :] = pg_reg.A

    # Record iteration time
    t1 = time.time()
    iteration_times_pg[t] = t1 - t0
  
  print("Done.")

  # Convert to torch
  thetas_pg = torch.from_numpy(thetas_pg)

  avg_iteration_times_pg[rep] = iteration_times_pg.mean()
  mses_pg[rep] = (thetas_pg - thetas_pg.mean(0)).pow(2).mean().item()

  print("MSE:", mses_pg[rep].item())
  print("avg iteration time:", avg_iteration_times_pg[rep].item())
  print("-----------------------")


###
# Plots
###
iterations = torch.arange(0, n_reps)
samples_and_dimensions = list(zip(dimensions_list, samples_list))

linewidth = 3
markersize = 8
alpha = .8

red_color = (0.86, 0.3712, 0.33999999999999997)
blue_color = (0.33999999999999997, 0.43879999999999986, 0.86)
green_color = (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
purple_color = (0.5803921568627451, 0.403921568627451, 0.7411764705882353)

###
# Plot iteration times
###
plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

plt.plot(avg_iteration_times_pg.cpu().numpy(), 
         '-', alpha = alpha, marker="p", markersize=markersize, color=red_color, label="Polya-Gamma", linewidth = linewidth)

plt.plot(avg_iteration_times_hmc.cpu().numpy(), 
         '-', alpha = alpha, marker="p", markersize=markersize, color=green_color, label="HMC", linewidth = linewidth)

plt.plot(avg_iteration_times.cpu().numpy(), 
         '-', alpha = alpha, marker="v", markersize=markersize, color=blue_color, label="MHI", linewidth = linewidth)


plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Average iteration time (seconds)", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("avg_iteration_times_plot.png", pad_inches=0, bbox_inches='tight',)

###
# Plot MSE's
###
plt.clf()
plt.style.use("ggplot")
plt.figure(figsize=(10, 8))

plt.plot(mses_pg.cpu().numpy(), 
         '-', alpha = alpha, marker="p", markersize=markersize, color=red_color, label="Polya-Gamma", linewidth = linewidth)

plt.plot(mses_hmc.cpu().numpy(), 
         '-', alpha = alpha, marker="p", markersize=markersize, color=green_color, label="HMC", linewidth = linewidth)

plt.plot(mses.cpu().numpy(), 
         '-', alpha = alpha, marker="v", markersize=markersize, color=blue_color, label="MHI", linewidth = linewidth)


plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xticks(iterations, samples_and_dimensions)
plt.xlabel(r"The dimension and sample size: d, n", fontsize = 25, color="black")
plt.ylabel(r"Mean squared error", fontsize = 25, color="black")
plt.legend(loc="best", fontsize=25, borderpad=.05, framealpha=0)
plt.savefig("mses_plot.png", pad_inches=0, bbox_inches='tight',)



