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
from torchbiggraph.types import LongTensorType


class EdgeList:
    @classmethod
    def empty(cls) -> "EdgeList":
        return cls(
            EntityList.empty(), EntityList.empty(), torch.empty((0,), dtype=torch.long)
        )

    @classmethod
    def cat(cls, edge_lists: Sequence["EdgeList"]) -> "EdgeList":
        cat_lhs = EntityList.cat([el.lhs for el in edge_lists])
        cat_rhs = EntityList.cat([el.rhs for el in edge_lists])

        if any(el.has_weight() for el in edge_lists):
            if not all(el.has_weight() for el in edge_lists):
                raise RuntimeError(
                    "Can't concatenate edgelists with and without weight field."
                )
            cat_weight = torch.cat([el.weight.expand((len(el),)) for el in edge_lists])
        else:
            cat_weight = None

        if any(el.has_time() for el in edge_lists):
            if not all(el.has_time() for el in edge_lists):
                raise RuntimeError(
                    "Can't concatenate edgelists with and without weight field."
                )
            cat_time = torch.cat([el.time.expand((len(el),)) for el in edge_lists])
        else:
            cat_time = None

        if all(el.has_scalar_relation_type() for el in edge_lists):
            rel_types = {el.get_relation_type_as_scalar() for el in edge_lists}
            if len(rel_types) == 1:
                (rel_type,) = rel_types
                return cls(
                    cat_lhs,
                    cat_rhs,
                    torch.tensor(rel_type, dtype=torch.long),
                    cat_weight,
                    cat_time,
                )
        cat_rel = torch.cat([el.rel.expand((len(el),)) for el in edge_lists])

        return cls(cat_lhs, cat_rhs, cat_rel, cat_weight, cat_time)

    def __init__(
        self,
        lhs: EntityList,
        rhs: EntityList,
        rel: LongTensorType,
        weight: Optional[LongTensorType] = None,
        time: Optional[LongTensorType] = None,
    ) -> None:
        if not isinstance(lhs, EntityList) or not isinstance(rhs, EntityList):
            raise TypeError(
                "Expected left- and right-hand side to be entity lists, got "
                "%s and %s instead" % (type(lhs), type(rhs))
            )
        if not isinstance(rel, (torch.LongTensor, torch.cuda.LongTensor)):
            raise TypeError("Expected relation to be a long tensor, got %s" % type(rel))
        if len(lhs) != len(rhs):
            raise ValueError(
                "The left- and right-hand side entity lists have different "
                "lengths: %d != %d" % (len(lhs), len(rhs))
            )
        if rel.dim() > 1:
            raise ValueError(
                "The relation can be either a scalar or a 1-dimensional "
                "tensor, got a %d-dimensional tensor" % rel.dim()
            )
        if rel.dim() == 1 and rel.shape[0] != len(lhs):
            raise ValueError(
                "The relation has a different length than the entity lists: "
                "%d != %d" % (rel.shape[0], len(lhs))
            )

        if weight is not None and (weight.nelement() == 0):
            weight = None

        if weight is not None:
            if weight.dim() > 1:
                raise ValueError(
                    "The weight can be either a scalar or a 1-dimensional "
                    "tensor, got a %d-dimensional tensor" % weight.dim()
                )
            if weight.dim() == 1 and weight.shape[0] != len(lhs):
                raise ValueError(
                    "The weight has a different length than the entity lists: "
                    "%d != %d" % (weight.shape[0], len(lhs))
                )
        if time is not None and (time.nelement() == 0):
            time = None

        if time is not None:
            if time.dim() > 1:
                raise ValueError(
                    "The time can be either a scalar or a 1-dimensional "
                    "tensor, got a %d-dimensional tensor" % weight.dim()
                )
            if time.dim() == 1 and time.shape[0] != len(lhs):
                raise ValueError(
                    "The time has a different length than the entity lists: "
                    "%d != %d" % (weight.shape[0], len(lhs))
                )

        self.lhs = lhs
        self.rhs = rhs
        self.rel = rel
        self.weight = weight
        self.time = time

    def has_scalar_relation_type(self) -> bool:
        return self.rel.dim() == 0

    def get_relation_type_as_scalar(self) -> int:
        if self.rel.dim() != 0:
            raise RuntimeError("The relation isn't a scalar")
        return int(self.rel)

    def get_relation_type_as_vector(self) -> LongTensorType:
        if self.rel.dim() == 0:
            return self.rel.view((1,)).expand((len(self),))
        return self.rel

    def get_relation_type(self) -> Union[int, LongTensorType]:
        if self.has_scalar_relation_type():
            return self.get_relation_type_as_scalar()
        else:
            return self.get_relation_type_as_vector()

    def has_time(self) -> bool:
        return self.time is not None

    def has_weight(self) -> bool:
        return self.weight is not None

    def get_weight(self) -> Union[float, torch.Tensor, None]:
        if self.has_time():
            return self.time
        else:
            return None

    def get_weight(self) -> Union[float, torch.Tensor]:
        if self.has_weight():
            return self.weight
        else:
            return 1

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EdgeList):
            return NotImplemented
        return (
            self.lhs == other.lhs
            and self.rhs == other.rhs
            and torch.equal(self.rel, other.rel)
        )

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return "EdgeList(%r, %r, %r, %r, %r)" % (
                self.lhs, self.rhs, self.rel, self.weight, self.time
        )

    def __getitem__(self, index: Union[int, slice, LongTensorType]) -> "EdgeList":
        if not isinstance(
            index, (int, slice, (torch.LongTensor, torch.cuda.LongTensor))
        ):
            raise TypeError(
                "Index can only be int, slice or long tensor, got %s" % type(index)
            )
        if (
            isinstance(index, (torch.LongTensor, torch.cuda.LongTensor))
            and index.dim() != 1
        ):
            raise ValueError(
                "Long tensor index must be 1-dimensional, got %d-dimensional"
                % (index.dim(),)
            )
        sub_lhs = self.lhs[index]
        sub_rhs = self.rhs[index]
        if self.has_scalar_relation_type():
            sub_rel = self.rel
        else:
            sub_rel = self.rel[index]
        if self.has_weight():
            sub_weight = self.weight[index]
        else:
            sub_weight = None
        if self.has_time():
            sub_time = self.time[index]
        else:
            sub_time = None
        return type(self)(sub_lhs, sub_rhs, sub_rel, sub_weight, sub_time)

    def __len__(self) -> int:
        return len(self.lhs)

    def to(self, *args, **kwargs):
        return type(self)(
            self.lhs.to(*args, **kwargs),
            self.rhs.to(*args, **kwargs),
            self.rel.to(*args, **kwargs),
        )
