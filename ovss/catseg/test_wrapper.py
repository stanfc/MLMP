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


def test_forward_shape():
    """Phase 2: forward produces the right output shape."""
    print("\n=== Test: forward shape contract ===")
    from ovss.catseg import load_catseg
    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')

    B, T, C = 1, 1, 20
    x = torch.randn(B, 3, 224, 224)
    text_x = torch.randn(T, C, 768)

    # interpolate=True -> output matches input spatial dims
    logits_full, img_feats, text_feats = wrapper(x, text_x,
                                                  text_ensemble=True,
                                                  interpolate=True)
    check(f"logits(interpolate=True) shape == (T,B,C,224,224), got {tuple(logits_full.shape)}",
          tuple(logits_full.shape) == (T, B, C, 224, 224))

    # interpolate=False -> output stays at aggregator grid (27x27)
    logits_grid, _, _ = wrapper(x, text_x,
                                 text_ensemble=True,
                                 interpolate=False)
    check(f"logits(interpolate=False) shape == (T,B,C,27,27), got {tuple(logits_grid.shape)}",
          tuple(logits_grid.shape) == (T, B, C, 27, 27))


def test_gradient_flow():
    """Phase 2: backward through forward updates ONLY visual LN params."""
    print("\n=== Test: gradient flow restricted to visual LN ===")
    from ovss.catseg import load_catseg
    wrapper, _ = load_catseg(backbone='ViT-L/14', device='cpu')

    # Mark trainable params using the same convention as tent_continual.py
    wrapper.transformer.requires_grad_(False)
    wrapper.ln_final.requires_grad_(False)
    wrapper.token_embedding.requires_grad_(False)
    for m in wrapper.visual.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
        else:
            for p in m.parameters(recurse=False):
                p.requires_grad_(False)

    # Snapshot aggregator + a non-LN visual param
    agg_snapshot = {
        n: p.detach().clone()
        for n, p in wrapper.aggregator.named_parameters()
    }
    conv1_snapshot = wrapper.visual.conv1.weight.detach().clone()

    # Pick one visual LN param to verify it actually updates
    target_ln_name = None
    target_ln_snapshot = None
    for n, p in wrapper.visual.named_parameters():
        if 'ln' in n.lower() and p.requires_grad:
            target_ln_name = n
            target_ln_snapshot = p.detach().clone()
            break
    check("found at least one trainable visual LN param",
          target_ln_name is not None)

    # Run one optimizer step
    params = [p for p in wrapper.visual.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    x = torch.randn(1, 3, 224, 224)
    text_x = torch.randn(1, 20, 768)
    logits, _, _ = wrapper(x, text_x, text_ensemble=True, interpolate=False)
    loss = -(logits.softmax(2) * logits.log_softmax(2)).sum(2).mean()
    loss.backward()
    opt.step()

    # Aggregator unchanged
    for n, p_after in wrapper.aggregator.named_parameters():
        check(f"aggregator.{n} unchanged",
              torch.allclose(p_after.detach(), agg_snapshot[n]))
    # Non-LN visual param unchanged
    check("visual.conv1.weight unchanged",
          torch.allclose(wrapper.visual.conv1.weight.detach(), conv1_snapshot))
    # Trainable LN param did update
    target_now = dict(wrapper.visual.named_parameters())[target_ln_name].detach()
    check(f"trainable LN param {target_ln_name} updated",
          not torch.allclose(target_now, target_ln_snapshot))


if __name__ == "__main__":
    test_import_and_instantiate()
    test_load_ovss_routing()
    test_forward_shape()
    test_gradient_flow()
    if _failed:
        print("\n*** SANITY CHECKS FAILED ***")
        sys.exit(1)
    print("\n*** ALL CHECKS PASSED ***")
