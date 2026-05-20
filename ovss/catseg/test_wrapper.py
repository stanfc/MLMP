"""Standalone sanity checks for CATSegWrapper.

Run: python -m ovss.catseg.test_wrapper

Exits non-zero on assertion failure. Print-driven, no pytest dependency.
Each phase of the implementation plan adds checks here.
"""
import sys
import torch
import torch.nn as nn


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    if not cond:
        global _failed
        _failed = True


_failed = False


def test_import_and_instantiate():
    """Phase 1: Package imports work and wrapper instantiates with stub."""
    print("\n=== Test: import and instantiate ===")
    from ovss.catseg import load_catseg
    wrapper, tokenize = load_catseg(backbone='ViT-L/14', device='cpu')
    check("load_catseg returns wrapper", wrapper is not None)
    check("load_catseg returns callable tokenizer", callable(tokenize))
    check("wrapper.visual is nn.Module",
          isinstance(wrapper.visual, nn.Module))
    check("wrapper.transformer is nn.Module",
          isinstance(wrapper.transformer, nn.Module))
    check("wrapper.ln_final is nn.LayerNorm",
          isinstance(wrapper.ln_final, nn.LayerNorm))
    check("wrapper.token_embedding is nn.Embedding",
          isinstance(wrapper.token_embedding, nn.Embedding))
    check("wrapper.logit_scale is nn.Parameter",
          isinstance(wrapper.logit_scale, nn.Parameter))
    check("wrapper has aggregator attribute",
          hasattr(wrapper, 'aggregator'))
    check("wrapper.aggregator is nn.Module",
          isinstance(wrapper.aggregator, nn.Module))
    # Trainable-param invariant: all aggregator params frozen
    agg_grads = [p.requires_grad for p in wrapper.aggregator.parameters()]
    check("aggregator all params frozen",
          len(agg_grads) == 0 or not any(agg_grads))
    check("aggregator in eval mode", not wrapper.aggregator.training)
    return wrapper


def test_load_ovss_routing():
    """Phase 1: load_ovss('catseg', ...) routes to CATSegWrapper."""
    print("\n=== Test: load_ovss routing ===")
    from ovss import load_ovss
    from ovss.catseg.catseg_wrapper import CATSegWrapper
    model, tokenize = load_ovss('catseg', 'ViT-L/14', device='cpu')
    check("load_ovss('catseg', ...) returns CATSegWrapper",
          isinstance(model, CATSegWrapper))
    check("tokenizer is callable", callable(tokenize))


if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
