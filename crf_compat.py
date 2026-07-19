"""
Compatibility shim for the `TorchCRF` package.

The project code was written against a CRF API that exposes:

    CRF(num_labels, batch_first=True)
    crf(emissions, labels, mask=..., reduction='mean')   # -> scalar loss term
    crf.decode(emissions, mask=...)                       # -> list of tag sequences

The current `torchcrf` (PyPI: TorchCRF) 1.1.0 API differs:

    CRF.__init__(num_labels, pad_idx=None, use_gpu=True)   # no batch_first
    CRF.forward(h, labels, mask)  -> per-sample LOG-likelihood (<= 0)
    CRF.viterbi_decode(h, mask)   # instead of .decode()

TorchCRF's internal layout is already (batch, seq, num_labels) i.e. batch-first,
so the only work needed is: accept `batch_first`, provide `.decode()`, and reduce
the log-likelihood to a scalar. Because `losses.py` does `crf_loss = -crf(...)`,
returning the (negative) mean log-likelihood makes that negation a positive mean
NLL loss, which is the standard CRF loss.
"""
import torch
import torch.nn as nn

try:
    from TorchCRF import CRF as _BaseCRF
except ImportError:  # case-insensitive FS / alternate install name
    from torchcrf import CRF as _BaseCRF


class CRF(_BaseCRF):
    def __init__(self, num_labels, batch_first=True, pad_idx=None, use_gpu=True):
        # TorchCRF manages its own device via `use_gpu`; we disable its internal
        # .cuda() juggling and rely on the parent model's device instead.
        super().__init__(num_labels, pad_idx=pad_idx, use_gpu=False)
        self.batch_first = batch_first

    @staticmethod
    def _full_mask(emissions, mask):
        if mask is None:
            return torch.ones(
                emissions.size(0), emissions.size(1),
                dtype=torch.bool, device=emissions.device
            )
        return mask

    def forward(self, emissions, labels, mask=None, reduction="mean"):
        mask = self._full_mask(emissions, mask)
        # per-sample log-likelihood (<= 0)
        ll = super().forward(emissions, labels, mask)
        if reduction in (None, "none"):
            return ll
        if reduction == "sum":
            return ll.sum()
        return ll.mean()  # negative scalar; losses.py negates -> positive NLL

    def decode(self, emissions, mask=None):
        mask = self._full_mask(emissions, mask)
        return self.viterbi_decode(emissions, mask)
