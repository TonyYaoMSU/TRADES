import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable #autograd oops
import torch.optim as optim

# core code of TRADES
# shhhhhh where does the author use l2-norm?????

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()



# core function for TRADES calculating traded_loss
def trades_loss(model,                  
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,   # the coeff of second term
                distance='l_inf'):
    # define KL-loss for inner maximization https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html?highlight=kldivloss#torch.nn.KLDivLoss
    # If the field size_average is set to False, the losses are instead summed for each minibatch.
    criterion_kl = nn.KLDivLoss(size_average=False)   
    # how to use loss :   f_loss(*args)(input) <- two parenthesis
    #eval() for BN and Dropout
    model.eval()    

    # feed x_natural here into the loss as a batch
    batch_size = len(x_natural)

    # generate adversarial example
    # initiate an x_adv for skipping the concave
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cpu().detach()   
    # detach() tensor won't give it grad calculations anymore.
    if distance == 'l_inf':  # L-infinity ball  # no random start here
        for _ in range(perturb_steps):  # FGSM_k
            x_adv.requires_grad_()         # start from x_adv 
            with torch.enable_grad():   # enable_grad vs no_grad 
            # For the maximization problem, using torch.nn.KLDivLoss and cross entropy is equivalent because they differ by a constant additive term.

                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))  # why first term log while second term origin: because in the loss_criteria, there is no "log_target = True"
                            
            grad = torch.autograd.grad(loss_kl, [x_adv])[0] # Computes and returns the sum of gradients of outputs w.r.t. the inputs.
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            #clamp .. 
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)

            #clamp original pic
            x_adv = torch.clamp(x_adv, 0.0, 1.0)


    elif distance == 'l_2':# L_2  we will come back later about l_2....not commented yet

        delta = 0.001 * torch.randn(x_natural.shape).cpu().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)

    # not implemented for other losses
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # adding two losses: L(fx,y)      ,     L(fx',fx)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)



    # not the main part, code related only
    # zero gradient     again, zero_grad -> loss_back -> updae?
    optimizer.zero_grad()





    # calculate robust loss
    logits = model(x_natural)      # pred of fx
    loss_natural = F.cross_entropy(logits, y)   # loss of  fx,y
    # loss of fx' fx
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                    F.softmax(model(x_natural), dim=1))
    loss = loss_natural + beta * loss_robust
    return loss
