"""
RDumb-Continual (Press et al., NeurIPS 2023, arXiv 2306.05401): TENT-style
entropy minimization + a FIXED-INTERVAL full reset to the source weights.
Every `reset_interval` adaptation calls (= images at batch_size 1) the model AND
optimizer are reset to the source state. Non-adaptive by design (that is the point
of the baseline). Subclass of TENTContinual: loss/optimizer identical to our TENT
baseline (single prompt, pixel-wise entropy, Adam on visual LayerNorm).
NOTE: violates the strict no-reset continual protocol; report it as a
periodic-reset reference.
"""
from .tent_continual import TENTContinual


class RDumbContinual(TENTContinual):
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None,
                 maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu',
                 reset_interval=1000):
        super().__init__(ovss_type=ovss_type, ovss_backbone=ovss_backbone, lr=lr, classes=classes,
                         steps=steps, catseg_checkpoint=catseg_checkpoint,
                         maft_checkpoint=maft_checkpoint, prompt_dir=prompt_dir,
                         runtime_calculation=runtime_calculation, device=device)
        self.reset_interval = int(reset_interval)
        self.n_adapt_calls = 0
        self.n_resets = 0
        print(f"+++ RDumb: full reset to source every {self.reset_interval} adaptation calls")

    def continual_adapt(self, x):
        out = self.perform_adaptation(x)
        self.n_adapt_calls += 1
        if self.reset_interval > 0 and self.n_adapt_calls % self.reset_interval == 0:
            self.reset()
            self.n_resets += 1
        return out

    def adapt(self, x):
        return self.continual_adapt(x)

    def state_dict(self):
        return {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'n_adapt_calls': self.n_adapt_calls, 'n_resets': self.n_resets}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.n_adapt_calls = state.get('n_adapt_calls', 0)
        self.n_resets = state.get('n_resets', 0)
