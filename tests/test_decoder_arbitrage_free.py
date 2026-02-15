from __future__ import annotations

import numpy as np
import torch

from spygen.models.decoder import ArbitrageFreeDecoder
from spygen.surface.arb_checks import is_arb_free


def test_decoder_outputs_arb_free_surface() -> None:
    torch.manual_seed(0)
    decoder = ArbitrageFreeDecoder(nx=21, nt=6)
    raw = torch.randn(8, decoder.param_dim)
    surface = decoder(raw).detach().numpy()
    for i in range(surface.shape[0]):
        assert is_arb_free(np.asarray(surface[i], dtype=float), tol=1e-6)
