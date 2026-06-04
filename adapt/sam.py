"""
Sharpness-Aware Minimization (SAM) optimizer wrapper.

Wraps a base optimizer with two-step SAM updates: first perturb weights to a
worst-case nearby point, then take the real gradient step from there.

Used by adapt/sar_continual.py to find flatter minima of the entropy loss.
Reference: Foret et al., "Sharpness-Aware Minimization", ICLR 2021.
"""

import torch


class SAM(torch.optim.Optimizer):
    """Two-step SAM. Call first_step() after first backward, then second
    backward and second_step()."""

    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)                            # climb to perturbed w + e(w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])           # restore to original w
        self.base_optimizer.step()                     # real update with grad at perturbed point
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.zeros(1, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
