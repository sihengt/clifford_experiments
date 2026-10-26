import torch


def gausLogLikelihood(predMean, predVar, groundTruth, multiVariate=True):
    """
    Log-likelihood of groundTruth under a Gaussian. With multiVariate=True, predVar is the Cholesky factor
    of the covariance; otherwise it is the standard deviation of each dimension.
    """
    if multiVariate:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(predMean,scale_tril=predVar)
        return dist.log_prob(groundTruth)
    else:
        dist = torch.distributions.normal.Normal(predMean,predVar)
        return dist.log_prob(groundTruth)
