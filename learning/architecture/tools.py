import torch

def batchMatMul(tensorA,tensorB):
    return torch.bmm(tensorA.reshape(-1,*tensorA.shape[-2:]),
                    tensorB.reshape(-1,*tensorB.shape[-2:]))\
                    .reshape(*tensorA.shape[:-1],tensorB.shape[-1])

def gausLogLikelihood(predMean,predVar,groundTruth,multiVariate=True):
    if multiVariate:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(predMean,scale_tril=predVar)
        return dist.log_prob(groundTruth)
    else:
        dist = torch.distributions.normal.Normal(predMean,predVar)
        return dist.log_prob(groundTruth)

def sampleGaus(predMean,predVar,multiVariate=True):
    #normSample = torch.randn(predMean.shape,device = predMean.device)
    #return batchMatMul(predVar,normSample.unsqueeze(-1)).squeeze(-1) + predMean
    if multiVariate:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(predMean,scale_tril=predVar)
        return dist.sample()
    else:
        dist = torch.distributions.normal.Normal(predMean,predVar)
        return dist.sample()

def sampleGaus2(predMean,predVar,multiVariate=True):
    if multiVariate:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(predMean,scale_tril=predVar)
        return dist.rsample()
    else:
        dist = torch.distributions.normal.Normal(predMean,predVar)
        return dist.rsample()

def combineParticleLogLikes(logLikes,pDim=1):
    # check to make sure this function is right
    maxLogLike = logLikes.max(dim=pDim,keepdim=True)[0]
    normalizedLogLikes = logLikes - maxLogLike
    logMeanLike = normalizedLogLikes.exp().mean(dim=pDim).log()+maxLogLike
    return logMeanLike
