import torch
import torch.nn.functional as F

@torch.no_grad()
def clamp_(x, min_val=0.0, max_val=1.0):
    return torch.clamp(x, min_val, max_val)



def fgsm_attack(model, x, y, eps, loss_fn):
    """
    Untargeted FGSM for binary classifier with BCEWithLogitsLoss.
    x in [0,1].
    """
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    logits = model(x_adv).squeeze(1)
    loss = loss_fn(logits, y)
    loss.backward()

    grad_sign = x_adv.grad.detach().sign()
    x_adv = x_adv + eps * grad_sign
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    return x_adv


def pgd_attack(model, x, y, eps, alpha, steps, loss_fn, random_start=True):
    """
    Untargeted PGD (L_inf) attack.
    - eps: max perturbation (L_inf ball)
    - alpha: step size
    - steps: number of iterations
    - random_start: start from random point in eps-ball
    """
    x0 = x.detach()

    if random_start:
        # uniform random in [-eps, eps]
        x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x0.clone()

    for _ in range(int(steps)):
        x_adv.requires_grad_(True)
        logits = model(x_adv).squeeze(1)
        loss = loss_fn(logits, y)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        # Project back to L_inf ball around x0
        delta = torch.clamp(x_adv - x0, min=-eps, max=eps)
        x_adv = torch.clamp(x0 + delta, 0.0, 1.0).detach()

    return x_adv
