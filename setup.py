# Update
# python3 setup.py sdist
# twine upload dist/*

from setuptools import setup, find_packages

setup(
  name="cmhi", 
  version="0.3",
  author="Austin Brown",
  author_email="brow5079@umn.edu",
  description="Centered Metropolis-Hastings independence algorithm for Bayesian logistic regression",
  url = "https://github.com/austindavidbrown/Centered-Metropolis-Hastings",
  packages=["cmhi"],
  install_requires=["torch >= 1.9.1"],
  keywords=["Metropolis-Hastings", "MCMC"]
)