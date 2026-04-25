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