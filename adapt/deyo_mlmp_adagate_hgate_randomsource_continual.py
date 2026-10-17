"""
DeYOMLMPAdaGateHGateRandomSourceContinual -- ablation: fully uninformed restore.

Subclass of the random-restore ablation (coin-flip trigger p=fire_prob). On top of
the random trigger, the restore TARGET is also uninformed: every triggered restore
pulls per-element (prob current_rst=base_rst) toward the SOURCE LN weights (the
weights at initialisation), never toward best_snapshot / lag-deque anchors. So
neither WHEN (grad-norm trend) nor WHERE TO (grad_norm/H_margin-chosen anchor) is
informed by any monitoring signal. Separate method name/save_dir; does not touch
CAGA's own results.
"""
import torch

from .deyo_mlmp_adagate_hgate_randomrestore_continual import DeYOMLMPAdaGateHGateRandomRestoreContinual


class DeYOMLMPAdaGateHGateRandomSourceContinual(DeYOMLMPAdaGateHGateRandomRestoreContinual):
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None,
                 maft_checkpoint=None,
                 vision_outputs=tuple(range(-1, -19, -1)),
                 prompt_dir='prompts.yaml',
                 deyo_margin_factor=0.5,
                 deyo_margin_e0_factor=0.4,
                 plpd_threshold=0.2,
                 aug_type='patch',
                 patch_len=4,
                 reweight_ent=True,
                 reweight_plpd=True,
                 top_block_exclude=6,
                 slope_window=10,
                 slope_deadzone=0.002,
                 lag_gain=1500.0,
                 base_rst=0.01,
                 max_windows=2000,
                 h_drop_ratio=0.9,
                 h_max_mode='lifetime',
                 h_max_decay=0.98,
                 maxlag_shallow=6,
                 monitor_interval=50,
                 trend_stat='mad',
                 trend_thr=0.5,
                 trend_hist=50,
                 lag_mode='ecdf',
                 lag_sat=1.5,
                 shallow_cap_mode='fixed',
                 anchor_h_min_ratio=0.95,
                 fire_prob=0.296,
                 random_seed=0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        super().__init__(
            ovss_type=ovss_type, ovss_backbone=ovss_backbone, lr=lr, classes=classes,
            steps=steps, catseg_checkpoint=catseg_checkpoint, maft_checkpoint=maft_checkpoint,
            vision_outputs=vision_outputs, prompt_dir=prompt_dir,
            deyo_margin_factor=deyo_margin_factor, deyo_margin_e0_factor=deyo_margin_e0_factor,
            plpd_threshold=plpd_threshold, aug_type=aug_type, patch_len=patch_len,
            reweight_ent=reweight_ent, reweight_plpd=reweight_plpd,
            top_block_exclude=top_block_exclude, slope_window=slope_window,
            slope_deadzone=slope_deadzone, lag_gain=lag_gain, base_rst=base_rst,
            max_windows=max_windows, h_drop_ratio=h_drop_ratio, h_max_mode=h_max_mode,
            h_max_decay=h_max_decay, maxlag_shallow=maxlag_shallow,
            monitor_interval=monitor_interval, trend_stat=trend_stat, trend_thr=trend_thr,
            trend_hist=trend_hist, lag_mode=lag_mode, lag_sat=lag_sat,
            shallow_cap_mode=shallow_cap_mode, anchor_h_min_ratio=anchor_h_min_ratio,
            fire_prob=fire_prob, random_seed=random_seed,
            save_dir=save_dir, runtime_calculation=runtime_calculation, device=device)
        # source LN weights = window-buffer entry 0 at construction (never evicted here)
        self._src_snapshot = self._win_buf[0]
        print("+++ CAGA-hgate-RANDOM-SOURCE: coin-flip trigger AND restore target = source weights")

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self._src_snapshot[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)
