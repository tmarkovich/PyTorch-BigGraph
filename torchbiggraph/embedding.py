#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from typing import Any, Sequence, Union
from typing import Optional

import torch
from torchbiggraph.entitylist import EntityList
from torchbiggraph.types import FloatTensorType

class Embeddings:
    """Serves as a wrapper of the embeddings.
    
    self.embs is the set of embedding vectors for each entity
    self.temporal_weights is the set of temporals weights for each entity
    self.temporal_biases is the set of temporals biases for each entity
    """

    def __init__(self,
                 embs: FloatTensorType,
                 temporal_weights: Optional[FloatTensorType] = None,
                 temporal_biases: Optional[FloatTensorType] = None):

        if not isinstance(embs, (torch.FloatTensor, torch.cuda.FloatTensor)):
            raise TypeError(
                f"Expected float embs as first argument, got {type(embs)}"
            )
        
        if ( (temporal_weights is not None) and
            (not isinstance(temporal_weights, (torch.FloatTensor, torch.cuda.FloatTensor)))
            ):
            raise TypeError(
                f"Expected float temporal_weights as first argument, got {type(embs)}"
            )

        if ( (temporal_biases is not None) and
            (not isinstance(temporal_biases, (torch.FloatTensor, torch.cuda.FloatTensor)))
            ):
            raise TypeError(
                f"Expected float temporal_biases as first argument, got {type(embs)}"
            )

        self.embs = embs
        self.temporal_weights = temporal_weights
        self.temporal_biases = temporal_biases

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Embeddings):
            return NotImplemented
        return (
            torch.equal(self.embs, other.embs)
            and torch.equal(self.temporal_weights, other.temporal_weights)
            and torch.equal(self.temporal_biases, other.temporal_biases)
        )

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return "EntityList(%r, TensorList(%r, %r))" % (
            self.embs,
            self.temporal_weights,
            self.temporal_biases,
        )

    def __len__(self) -> int:
        return self.embs.shape[0]