import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


# pyannote model checkpoints contain non-tensor objects incompatible with
# torch 2.6+ default of weights_only=True
torch.load = _patched_torch_load  # ty: ignore[invalid-assignment]
