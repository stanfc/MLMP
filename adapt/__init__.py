import inspect
from pprint import pprint

from .mlmp import MLMP
from .clipartt import CLIPARTT
from .watt import WATT
from .tpt import TPT
from .tent import TENT
from .cotta import CoTTA
from .dpcore import DPCore

# Continual TTA (no reset, state persists across stream)
from .mlmp_continual import MLMPContinual
from .tent_continual import TENTContinual
from .cma_continual import CMAContinual
from .cma_proto_continual import CMAProtoContinual
from .cma_layered_continual import CMALayeredContinual
from .cma_divgate_continual import CMADivGateContinual
from .tent_divgate_continual import TENTDivGateContinual
from .tent_divgate_layered_continual import TENTDivGateLayeredContinual
from .tent_early_continual import TENTEarlyContinual
from .tent_contgate_continual import TENTContGateContinual
from .tent_siggate_continual import TENTSigGateContinual
from .tent_divgate_continual_catseg import TENTDivGateContinualCatSeg
from .kff import KFF
from .tent_divgate_recent_anchor import TENTDivGateRecentAnchor
from .tent_divgate_decoupled import TENTDivGateDecoupled
from .tent_divgate_hybrid_anchor import TENTDivGateHybridAnchor
from .tent_divgate_smooth_anchor import TENTDivGateSmoothAnchor
from .clipartt_siggate_continual import CLIPArTTSigGateContinual
from .clipartt_continual import CLIPArTTContinual
from .clipartt_divgate_continual import CLIPArTTDivGateContinual
from .tent_topk_continual import TENTTopKContinual
from .tent_minprompt_continual import TENTMinPromptContinual
from .tent_topk_divgate_continual import TENTTopKDivGateContinual
from .tent_minprompt_divgate_continual import TENTMinPromptDivGateContinual
from .mlmp_smooth_anchor_continual import MLMPSmoothAnchorContinual
from .mlmp_topk_continual import MLMPTopKContinual
from .mlmp_minprompt_continual import MLMPMinPromptContinual
from .mlmp_topk_minprompt_continual import MLMPTopKMinPromptContinual
from .mlmp_topk_divgate_continual import MLMPTopKDivGateContinual
from .mlmp_topk_smooth_anchor_continual import MLMPTopKSmoothAnchorContinual
from .mlmp_minprompt_divgate_continual import MLMPMinPromptDivGateContinual
from .mlmp_minprompt_smooth_anchor_continual import MLMPMinPromptSmoothAnchorContinual
from .mlmp_topk_minprompt_divgate_continual import MLMPTopKMinPromptDivGateContinual
from .mlmp_topk_minprompt_smooth_anchor_continual import MLMPTopKMinPromptSmoothAnchorContinual
from .mlmp_simple_eval_continual import MLMPSimpleEvalContinual
from .tent_uaml_eval_continual import TENTUAMLEvalContinual
from .sar_continual import SARContinual
from .sar_divgate_continual import SARDivGateContinual
from .delta_continual import DELTAContinual
from .deyo_continual import DeYOContinual
from .rotta_continual import RoTTAContinual
from .deyo_uaml_continual import DeYOUAMLContinual
from .deyo_mlmp_continual import DeYOMLMPContinual
from .eata_continual import EATAContinual
from .dat_continual import DATContinual
from .deyo_mlmp_divgate_continual import DeYOMLMPDivGateContinual
from .deyo_mlmp_smooth_anchor_continual import DeYOMLMPSmoothAnchorContinual
from .mlmp_divgate_continual import MLMPDivGateContinual
from .sar_mlmp_smooth_anchor_continual import SARMLMPSmoothAnchorContinual

# Map methods to their classes
METHOD_CLASSES = {
    # Episodic TTA
    'mlmp': MLMP,
    'clipartt': CLIPARTT,
    'watt': WATT,
    'tpt': TPT,
    'tent': TENT,
    'cotta': CoTTA,
    'dpcore': DPCore,
    # Continual TTA (naive: no reset)
    'mlmp_continual': MLMPContinual,
    'tent_continual': TENTContinual,
    # Continual TTA with cross-modal alignment loss (proposed Direction 1)
    'cma_continual': CMAContinual,
    # Continual TTA with CMA + source/target prototype memory bank (proposed D1+D2)
    'cma_proto_continual': CMAProtoContinual,
    # Continual TTA with CMA + layer-stratified stochastic restoration (Direction A)
    'cma_layered_continual': CMALayeredContinual,
    # Continual TTA with CMA + diversity-gated stochastic restoration (Direction B)
    'cma_divgate_continual': CMADivGateContinual,
    # Continual TTA with TENT + diversity-gated stochastic restoration (Direction B on TENT)
    'tent_divgate_continual': TENTDivGateContinual,
    # TENT-DivGate with hard freeze on late LN layers (Direction A x B combined)
    'tent_divgate_layered_continual': TENTDivGateLayeredContinual,
    # Plain TENT but only early-block LN trained, rest frozen (no gate, no restore)
    'tent_early_continual': TENTEarlyContinual,
    # TENT with continuous H_margin -> rst mapping (no discrete modes)
    'tent_contgate_continual': TENTContGateContinual,
    # TENT with sigmoid-shaped H_margin -> rst mapping (plateaus at both ends)
    'tent_siggate_continual': TENTSigGateContinual,
    # TENT-DivGate on CAT-Seg backbone (CTTA)
    'tent_divgate_continual_catseg': TENTDivGateContinualCatSeg,
    # KFF (Class-aware domain knowledge Fusion & Fission, NeurIPS 2025)
    'kff': KFF,
    # TENT-DivGate with sliding-window anchor (restoration target = N batches ago)
    'tent_divgate_recent_anchor': TENTDivGateRecentAnchor,
    # TENT-DivGate with decoupled marginal_buf_size and monitor_interval
    'tent_divgate_decoupled': TENTDivGateDecoupled,
    # TENT-DivGate with hybrid anchor: cautious->recent, brake->source
    'tent_divgate_hybrid_anchor': TENTDivGateHybridAnchor,
    # TENT-DivGate with continuous H_margin -> lag(H) anchor mapping
    'tent_divgate_smooth_anchor': TENTDivGateSmoothAnchor,
    # CLIPArTT (self-distillation) base loss + sigmoid diversity gate
    'clipartt_siggate_continual': CLIPArTTSigGateContinual,
    # CLIPArTT base loss only (continual, no gate / no restore)
    'clipartt_continual': CLIPArTTContinual,
    # CLIPArTT base + 3-tier discrete DivGate
    'clipartt_divgate_continual': CLIPArTTDivGateContinual,
    # TENT entropy on top-K% confidence pixels only (aggressive variant)
    'tent_topk_continual': TENTTopKContinual,
    # Multi-prompt min-margin loss (aggressive consensus self-training)
    'tent_minprompt_continual': TENTMinPromptContinual,
    # Top-K TENT + 3-tier DivGate
    'tent_topk_divgate_continual': TENTTopKDivGateContinual,
    # MinPrompt + 3-tier DivGate
    'tent_minprompt_divgate_continual': TENTMinPromptDivGateContinual,
    # MLMP base + continuous smooth-anchor gate
    'mlmp_smooth_anchor_continual': MLMPSmoothAnchorContinual,
    # MLMP pixel entropy on top-K% confidence pixels (optional ILE)
    'mlmp_topk_continual': MLMPTopKContinual,
    # MLMP multi-prompt min-margin loss (optional ILE)
    'mlmp_minprompt_continual': MLMPMinPromptContinual,
    # MLMP TopK + MinPrompt combined (spatial + prompt-view filtering)
    'mlmp_topk_minprompt_continual': MLMPTopKMinPromptContinual,
    # MLMP-TopK with DivGate / SmoothAnchor
    'mlmp_topk_divgate_continual': MLMPTopKDivGateContinual,
    'mlmp_topk_smooth_anchor_continual': MLMPTopKSmoothAnchorContinual,
    # MLMP-MinPrompt with DivGate / SmoothAnchor
    'mlmp_minprompt_divgate_continual': MLMPMinPromptDivGateContinual,
    'mlmp_minprompt_smooth_anchor_continual': MLMPMinPromptSmoothAnchorContinual,
    # MLMP-TopK-MinPrompt (combined) with DivGate / SmoothAnchor
    'mlmp_topk_minprompt_divgate_continual': MLMPTopKMinPromptDivGateContinual,
    'mlmp_topk_minprompt_smooth_anchor_continual': MLMPTopKMinPromptSmoothAnchorContinual,
    # MLMP adapt loss + TENT-style simple evaluate (ablation)
    'mlmp_simple_eval_continual': MLMPSimpleEvalContinual,
    # TENT adapt loss + UAML multi-layer evaluate (ablation)
    'tent_uaml_eval_continual': TENTUAMLEvalContinual,
    # SAR (Sharpness-Aware Reliable TTA, ICLR 2023): reliable-pixel filter
    # + SAM optimizer + EMA-triggered model recovery
    'sar_continual': SARContinual,
    # DeYO (Entropy is not Enough, ICLR 2024 spotlight): entropy + PLPD
    # (patch-shuffle disagreement) dual filter + reweighted entropy loss
    'deyo_continual': DeYOContinual,
    # RoTTA (Robust TTA in Dynamic Scenarios, CVPR 2023): CSTU
    # category-balanced memory bank + EMA teacher-student + timeliness
    'rotta_continual': RoTTAContinual,
    # DeYO adapt loss + MLMP multi-LAYER adaptation + UAML eval (single prompt)
    'deyo_uaml_continual': DeYOUAMLContinual,
    # DeYO adapt loss + MLMP multi-prompt + multi-layer + UAML eval (full)
    'deyo_mlmp_continual': DeYOMLMPContinual,
    # EATA (Efficient Anti-forgetting TTA, ICML 2022): reliable+non-redundant
    # sample selection + Fisher (EWC) anti-forgetting regularizer
    'eata_continual': EATAContinual,
    # DAT (Distribution-Aware Tuning, CVPR 2024): grad-ranked selective
    # param tuning (DSP/TRP) split by per-pixel uncertainty + EMA teacher
    'dat_continual': DATContinual,
    # DeYO + full MLMP (multi-prompt/layer + UAML) + DivGate restore
    'deyo_mlmp_divgate_continual': DeYOMLMPDivGateContinual,
    # DeYO + MLMP + SmoothAnchor (continuous lag(H) restore)
    'deyo_mlmp_smooth_anchor_continual': DeYOMLMPSmoothAnchorContinual,
    # Continual TTA with MLMP loss + diversity-gated stochastic restoration
    'mlmp_divgate_continual': MLMPDivGateContinual,
    # Continual TTA with SAR (SAM + reliable filter) + diversity-gated restore (DivGate replaces SAR recovery)
    'sar_divgate_continual': SARDivGateContinual,
    # SAR (SAM + reliable filter) + full MLMP (multi-prompt/layer + UAML eval) + smooth-anchor restore
    'sar_mlmp_smooth_anchor_continual': SARMLMPSmoothAnchorContinual,
    # Continual TTA with TENT + class-aware Dynamic Online re-weighting (DELTA DOT, ICLR 2023)
    'delta_continual': DELTAContinual,
}



# Main function
def get_method(args, device):
    if args.method not in METHOD_CLASSES:
        raise ValueError(f"Unknown method: {args.method}")

    method_class = METHOD_CLASSES[args.method]
    sig = inspect.signature(method_class.__init__)
    args_dict = vars(args)

    method_args = {}
    missing_required = []

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if name == 'device':
            method_args['device'] = device
        elif name in args_dict:
            # arg present in argparse → use it
            method_args[name] = args_dict[name]
        elif param.default is not inspect.Parameter.empty:
            # arg not in argparse but has a constructor default → skip (Python uses default)
            pass
        else:
            # truly required arg with no default and not in argparse
            missing_required.append(name)

    if missing_required:
        raise ValueError(
            f"Missing required arguments for {method_class.__name__}: {missing_required}. "
            f"Add them to argparse."
        )

    return method_class(**method_args)

# Function to summarize the 'classes' argument
def summarize_args(method_args):
    summarized_args = method_args.copy()
    if 'classes' in summarized_args:
        summarized_args['classes'] = len(summarized_args['classes'])  # Replace with total count
    return summarized_args