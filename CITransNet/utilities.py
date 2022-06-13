import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.optim as optim
from clippedAdam        import Adam
import matplotlib.pyplot as plt
from IPython import embed

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

def utility(valuations, allocation, pay):
    """ Given input valuation , payment  and allocation , computes utility
            Input params:
                valuation : [num_batches, num_agents, num_items]
                allocation: [num_batches, num_agents, num_items]
                pay       : [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
    """
    return (torch.sum(valuations*allocation, dim=-1) - pay)

def revenue(pay):
    """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
    return torch.mean(torch.sum(pay, dim=-1))
    

def misreportUtility(mechanism,batch_data,batchMisreports):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining all the misreported utilities

    
    """
    batchTrueValuations     = batch_data[0]
    batch_bidder_context       =  batch_data[1]    # (bs, n_bidder, d)
    batch_item_context         = batch_data[2]     # (bs, n_item, d)
    nAgent             = batchTrueValuations.shape[-2]
    nObjects           = batchTrueValuations.shape[-1]
    batchSize          = batchTrueValuations.shape[0]
    nbrInitializations = batchMisreports.shape[1]
    
    V  = batchTrueValuations.unsqueeze(1)
    V  = V.repeat(1,nbrInitializations, 1, 1)
    V  = V.unsqueeze(0)
    V  = V.repeat(nAgent, 1, 1, 1, 1) # (n_bidder, bs, n_init, n_bidder, n_item)

    M  = batchMisreports.unsqueeze(0)
    M  = M.repeat(nAgent,1, 1, 1, 1)


    mask1                                           = np.zeros((nAgent,nAgent,nObjects))
    mask1[np.arange(nAgent),np.arange(nAgent),:]    = 1.0
    mask2                                           = np.ones((nAgent,nAgent,nObjects))
    mask2                                           = mask2-mask1
    
    mask1       = (torch.tensor(mask1).float()).to(device)
    mask2       = (torch.tensor(mask2).float()).to(device)

    V  = V.permute(1, 2, 0, 3, 4) # (bs, n_init, n_bidder, n_bidder, n_item)
    M  = M.permute(1, 2, 0, 3, 4)

    tensor      =  M*mask1 + V*mask2

    tensor      = tensor.permute(2, 0, 1, 3, 4) # (n_bidder, bs, n_init, n_bidder, n_item)
    bidder_context   = batch_bidder_context.view(1, batchSize, 1, nAgent, -1).repeat(nAgent, 1, nbrInitializations, 1, 1)
    item_context     = batch_item_context.view(1, batchSize, 1, nObjects, -1).repeat(nAgent, 1, nbrInitializations, 1, 1)

    V  = V.permute(2, 0, 1, 3, 4)
    M  = M.permute(2, 0, 1, 3, 4)

    tensor = View(-1,nAgent, nObjects)(tensor)
    tensor = tensor.float()
    bidder_context = bidder_context.view(tensor.shape[0], nAgent, -1)
    item_context = item_context.view(tensor.shape[0], nObjects, -1)

    allocation, payment = mechanism((tensor, bidder_context, item_context))

    allocation    =  View(nAgent,batchSize,nbrInitializations,nAgent, nObjects)(allocation)
    payment       =  View(nAgent,batchSize,nbrInitializations,nAgent)(payment)

    advUtilities    = torch.sum(allocation*V, dim=-1)-payment

    advUtility      = advUtilities[np.arange(nAgent),:,:,np.arange(nAgent)]
    
    return(advUtility.permute(1, 2, 0))


def misreportOptimization(mechanism,batch, data, misreports, R, gamma, minimum=0, maximum=1):

    """ This function takes the valuation and misreport batches
        and R the number of optimization step and modifies the misreport array



        """
    localMisreports     = misreports[:]
    batchMisreports     = torch.tensor(misreports[batch]).to(device)
    batchTrueValuations = torch.tensor(data[0][batch]).to(device)
    batch_bidder_type = torch.tensor(data[1][batch]).to(device)
    batch_item_type = torch.tensor(data[2][batch]).to(device)
    batch_data = (batchTrueValuations, batch_bidder_type, batch_item_type)
    batchMisreports.requires_grad = True

    opt = Adam([batchMisreports], lr=gamma)

    for k in range(R):
        advU         = misreportUtility(mechanism,batch_data,batchMisreports)
        loss         =  -1*torch.sum(advU).to(device)
        loss.backward()
        opt.step(restricted= True, min=minimum, max=maximum)
        opt.zero_grad()

    mechanism.zero_grad()

    localMisreports[batch,:,:,:] = batchMisreports.cpu().detach().numpy()
    return(localMisreports)

def trueUtility(mechanism,batch_data,allocation=None, payment=None):
    
    """ This function takes the valuation batches
        and returns a tensor constaining the utilities

    """
    if allocation is None or payment is None:
        allocation, payment = mechanism(batch_data)
    batchTrueValuations = batch_data[0]
    return utility(batchTrueValuations, allocation, payment)


def regret(mechanism, batch_data, batchMisreports, allocation, payment):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining the regrets for each bidder and each batch
        

    """
    missReportUtilityAll = misreportUtility(mechanism,batch_data,batchMisreports)
    misReportUtilityMax  = torch.max(missReportUtilityAll, dim =1)[0]
    return(misReportUtilityMax-trueUtility(mechanism,batch_data, allocation, payment))


def loss(mechanism, lamb, rho, batch, data, misreports):
    """
    This function tackes a batch which is a numpy array of indices and computes 
    the loss function                                                             : loss 
    the average regret per agent which is a tensor of size [nAgent]               : rMean 
    the maximum regret among all batches and agenrs which is a tensor of size [1] : rMax
    the average payments which is a tensor of size [1]                            : -paymentLoss
    
    """
    batchMisreports     = torch.tensor(misreports[batch]).to(device)
    batchTrueValuations = torch.tensor(data[0][batch]).to(device)
    batch_bidder_type = torch.tensor(data[1][batch]).to(device)
    batch_item_type = torch.tensor(data[2][batch]).to(device)
    batch_data = (batchTrueValuations, batch_bidder_type, batch_item_type)
    allocation, payment = mechanism(batch_data)

    paymentLoss          = -torch.sum(payment)/batch.shape[0]
    
    r                    = F.relu(regret(mechanism,batch_data,batchMisreports, allocation, payment))
    rMean                = torch.mean(r, dim=0).to(device)

    rMax                 = torch.max(r).to(device)

    lagrangianLoss       = torch.sum(rMean*lamb)

    lagLoss              = (rho/2)*torch.sum(torch.pow(rMean,2))

    loss = paymentLoss +lagrangianLoss+lagLoss

    return(loss, rMean, rMax, -paymentLoss)
