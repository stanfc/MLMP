"""
DeYOMLMPAdaGateAllLNContinual -- the flagship, with every visual LayerNorm adapting.

Ablation of ONE design choice: which parameters are allowed to move.

The flagship (`deyo_mlmp_adagate_continual`, i.e. "Ours") runs with
`--top_block_exclude 6`, which freezes `ln_post` and resblocks 18-23 and leaves
**74 of the 100** visual LayerNorm params adapting. That setting came in with
學長's GDG-PA / smooth-anchor lineage and has never been ablated: of every run in
`save/`, 241 used `6` and the only runs at `0` are from this plug-and-play study.
It matters -- TENT + AdaGate scores 28.15 at 74 params and 22.48 at 100 -- so the
flagship's 31.92 needs the same check.

This subclass changes NOTHING but the exclusion rule: with `top_block_exclude <= 0`
nothing is excluded at all, giving exactly **100** params, the same set every
published baseline (`tent/mlmp/delta/sar_continual`) trains. Everything else --
the DeYO+MLMP objective, the MAD-z trigger, the ECDF lag, the H_margin regime
switch, the permanent best anchor -- is inherited unchanged from 學長's file,
which is not touched.

Why a subclass rather than just passing `--top_block_exclude 0` to the parent:
the parent's `_is_excluded` excludes `ln_post` unconditionally, so the parent at
`0` would give 98, not 100, and would not be comparable to the four plug-and-play
runs that are all at exactly 100.

Run it with the flagship's own command and only `--top_block_exclude 0` changed
(note LR 5e-6, not the 1e-5 the plug-and-play arms use):
    bash bash/ablation_allln.sh <gpu> [rounds]
"""
from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class DeYOMLMPAdaGateAllLNContinual(DeYOMLMPAdaGateContinual):

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        # top_block_exclude <= 0 -> nothing frozen, all 100 params adapt, matching
        # what every published continual baseline trains.
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)
