import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def sp_logits(logits, s, m, y, num_classes=10):
    mask = F.one_hot(y, num_classes=num_classes)
    alpha_p = (1 + m - logits) * mask
    alpha_n = (logits + m) * (1 - mask)
    logits = s * (alpha_p * (logits + m - 1) + alpha_n * (logits + m))
    return logits


def kl_loss(logits, adv_logits, y, s, n, criterion, batch_size):
    logits = F.softmax(s*logits, dim=1)
    adv_logits = F.softmax(s*adv_logits, dim=1)
    kl = criterion(torch.log(adv_logits), logits)
    loss = (1.0 / batch_size) * torch.sum(
        torch.sum(kl*(kl+n), dim=1))
    return loss


def spat_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                m=0.2,
                n=0.2,
                s=5
                ):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits = model(x_natural)
            adv_logits = model(x_adv)
            loss_kl = kl_loss(logits, adv_logits, y, s, n, criterion_kl, batch_size)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    adv_logits = model(x_adv)
    loss_natural = F.cross_entropy(sp_logits(logits, s, m, y), y)
    # loss_natural = F.cross_entropy(logits, y)
    # loss_adv = F.cross_entropy(adv_logits, y)

    loss_robust = kl_loss(logits, adv_logits, y, s, n, criterion_kl, batch_size)
    loss = loss_natural + beta * loss_robust

    return loss