#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Sequence, Union

import torch
import torch.nn as nn
from torchbiggraph.plugin import PluginRegistry
from torchbiggraph.types import FloatTensorType, LongTensorType, Side
from torchbiggraph.util import match_shape


class AbstractOperator(nn.Module, ABC):

    """Perform the same operation on many vectors.

    Given a tensor containing a set of vectors, perform the same operation on
    all of them, with a common set of parameters. The dimension of these vectors
    will be given at initialization (so that any parameter can be initialized).
    The input will be a tensor with at least one dimension. The last dimension
    will contain the vectors. The output is a tensor that will have the same
    size as the input.

    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        pass

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        raise NotImplementedError("Regularizer not implemented for this operator")

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        return embs.abs()


OPERATORS = PluginRegistry[AbstractOperator]()


@OPERATORS.register_as("RotH")
class RotH_Operator(AbstractOperator):
    def __init__(self, dim: int,  bias = True):
        super().__init__(dim, bias)
        self.bias = bias
        self.data_type = torch.float

        if self.bias == True:
            self.dim = self.dim - 1

        data = 2 * torch.rand((self.dim, ), dtype=self.data_type) - 1.0
        self.rot_diag = nn.Parameter(data, requires_grad=True)
        self.rel = nn.Parameter(torch.rand((self.dim, )), requires_grad=True)
        c_init = torch.ones((1, ), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rotation_queries(self, queries):
        rot_diag = self.rot_diag.to(device=queries.device).view(1, -1)

        return givens_rotations_n(rot_diag, queries)

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:

        if self.bias == True:
            bias_term = embeddings[..., :1]
            embeddings = embeddings[..., 1:]

        match_shape(embeddings, ..., self.dim)
        lhs_rot_e = self.get_rotation_queries(embeddings).view((-1, 1, self.dim))
        rel_bias = self.rel.to(device=embeddings.device).view(1, -1)
        c = self.c.to(device=embeddings.device).view(1, -1).expand(embeddings.shape[0], -1)
        c = F.softplus(c)
        lhs_rot_e = lhs_rot_e.view(-1, self.dim)
        rel_bias = expmap0(rel_bias, c)
        result = expmap0(lhs_rot_e, c)
        result = project(mobius_add(result, rel_bias, c), c)

        if self.bias == True:
            result = torch.cat([c, bias_term, result], dim=-1)

        return result

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.rel.to(device=operator_idxs.device).abs()


@OPERATORS.register_as("RefH")
class RefH_Operator(AbstractOperator):
    def __init__(self, dim: int,  bias = True):
        super().__init__(dim, bias)
        self.bias = bias
        self.data_type = torch.float

        if self.bias == True:
            self.dim = self.dim - 1

        data = 2 * torch.rand((self.dim, ), dtype=self.data_type) - 1.0
        self.ref_diag = nn.Parameter(data, requires_grad=True)

        self.rel = nn.Parameter(torch.rand((self.dim, )), requires_grad=True)

        c_init = torch.ones((1, ), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_reflection_queries(self, queries):
        ref_diag = self.ref_diag.to(device=queries.device).view(1, -1)

        return givens_reflection_n(ref_diag, queries)

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:

        if self.bias == True:
            bias_term = embeddings[..., :1]
            embeddings = embeddings[..., 1:]

        match_shape(embeddings, ..., self.dim)

        lhs_rot_e = self.get_reflection_queries(embeddings).view((-1, 1, self.dim))
        rel_bias = self.rel.to(device=embeddings.device).view(1, -1)
        c = self.c.to(device=embeddings.device).view(1, -1).expand(embeddings.shape[0], -1)
        c = F.softplus(c)

        lhs_rot_e = lhs_rot_e.view(-1, self.dim)

        rel_bias = expmap0(rel_bias, c)
        result = expmap0(lhs_rot_e, c)

        result = project(mobius_add(result, rel_bias, c), c)

        if self.bias == True:
            result = torch.cat([c, bias_term, result], dim=-1)

        return result

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.rel.to(device=operator_idxs.device).abs()


@OPERATORS.register_as("AttH")
class AttHyper_Operator(AbstractOperator):
    def __init__(self, dim: int,  bias = True):
        super().__init__(dim, bias)
        self.bias = bias
        self.data_type = torch.float

        if self.bias == True:
            self.dim = self.dim - 1

        data = 2 * torch.rand((self.dim, ), dtype=self.data_type) - 1.0
        self.ref_diag = nn.Parameter(data, requires_grad=True)


        data = 2 * torch.rand((self.dim, ), dtype=self.data_type) - 1.0
        self.rot_diag = nn.Parameter(data, requires_grad=True)


        data = torch.randn((self.dim, ), requires_grad=True)
        self.context_vec = nn.Parameter(data, requires_grad=True)

        self.act = nn.Softmax(dim=1)

        scale_data = torch.Tensor([1. / np.sqrt(self.dim)])
        self.scale = nn.Parameter(scale_data)
        self.scale.requires_grad = False

        self.rel = nn.Parameter(torch.rand((self.dim, )), requires_grad=True)

        c_init = torch.ones((1, ), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)


    def get_reflection_queries(self, queries):
        rot_diag = self.rot_diag.to(device=queries.device).view(1, -1)

        return givens_reflection_n(rot_diag, queries)

    def get_rotation_queries(self, queries):
        ref_diag = self.ref_diag.to(device=queries.device).view(1, -1)

        return givens_rotations_n(ref_diag, queries)

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:

        if self.bias == True:
            bias_term = embeddings[..., :1]
            embeddings = embeddings[..., 1:]

        match_shape(embeddings, ..., self.dim)

        lhs_ref_e = self.get_reflection_queries(embeddings).view((-1, 1, self.dim))
        lhs_rot_e = self.get_rotation_queries(embeddings).view((-1, 1, self.dim))
        att_matrix = self.context_vec.to(device=embeddings.device)
        rel_bias = self.rel.to(device=embeddings.device).view(1, -1)

        c = self.c.to(device=embeddings.device).view(1, -1).expand(embeddings.shape[0], -1)
        c = F.softplus(c)

        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=1)
        context_vec = att_matrix.view((-1, 1, self.dim))
        scale = self.scale.to(device=embeddings.device)
        att_weights = torch.sum(context_vec * cands * scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        result = torch.sum(att_weights * cands, dim=1)
        rel_bias = expmap0(rel_bias, c)
        result = expmap0(result, c)
        result = project(mobius_add(result, rel_bias, c), c)

        if self.bias == True:
            result = torch.cat([c, bias_term, result], dim=-1)
        return result

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.rel.to(device=operator_idxs.device).abs()



@OPERATORS.register_as("none")
class IdentityOperator(AbstractOperator):
    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return None


@OPERATORS.register_as("diagonal")
class DiagonalOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.diagonal = nn.Parameter(torch.ones((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return self.diagonal.to(device=embeddings.device) * embeddings

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.diagonal.abs()


@OPERATORS.register_as("translation")
class TranslationOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        return embeddings + self.translation.to(device=embeddings.device)

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return self.translation.abs()


@OPERATORS.register_as("linear")
class LinearOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        # We add a dimension so that matmul performs a matrix-vector product.
        return torch.matmul(
            self.linear_transformation.to(device=embeddings.device),
            embeddings.unsqueeze(-1),
        ).squeeze(-1)


@OPERATORS.register_as("affine")
class AffineOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.linear_transformation = nn.Parameter(torch.eye(self.dim))
        self.translation = nn.Parameter(torch.zeros((self.dim,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        # We add a dimension so that matmul performs a matrix-vector product.
        return torch.matmul(
            self.linear_transformation.to(device=embeddings.device),
            embeddings.unsqueeze(-1),
        ).squeeze(-1) + self.translation.to(device=embeddings.device)

    # FIXME This adapts from the pre-D14024710 format; remove eventually.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = "%slinear_transformation" % prefix
        old_param_key = "%srotation" % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = (
                state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@OPERATORS.register_as("complex_diagonal")
class ComplexDiagonalOperator(AbstractOperator):
    def __init__(self, dim: int):
        super().__init__(dim)
        if dim % 2 != 0:
            raise ValueError(
                "Need even dimension as 1st half is real "
                "and 2nd half is imaginary coordinates"
            )
        self.real = nn.Parameter(torch.ones((self.dim // 2,)))
        self.imag = nn.Parameter(torch.zeros((self.dim // 2,)))

    def forward(self, embeddings: FloatTensorType) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        real_a = embeddings[..., : self.dim // 2]
        imag_a = embeddings[..., self.dim // 2 :]
        real_b = self.real.to(device=embeddings.device)
        imag_b = self.imag.to(device=embeddings.device)
        prod = torch.empty_like(embeddings)
        prod[..., : self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2 :] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self) -> Optional[FloatTensorType]:
        return torch.sqrt(self.real**2 + self.imag**2)

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        assert embs.shape[-1] == self.dim
        real, imag = embs[..., : self.dim // 2], embs[..., self.dim // 2 :]
        return torch.sqrt(real**2 + imag**2)


class AbstractDynamicOperator(nn.Module, ABC):

    """Perform different operations on many vectors.

    The inputs are a tensor containing a set of vectors and another tensor
    specifying, for each vector, which operation to apply to it. The output has
    the same size as the first input and contains the outputs of the operations
    applied to the input vectors. The different operations are identified by
    integers in a [0, N) range. They are all of the same type (say, translation)
    but each one has its own set of parameters. The dimension of the vectors and
    the total number of operations that need to be supported are provided at
    initialization. The first tensor can have any number of dimensions (>= 1).

    """

    def __init__(self, dim: int, num_operations: int):
        super().__init__()
        self.dim = dim
        self.num_operations = num_operations

    @abstractmethod
    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        pass

    def get_operator_params_for_reg(
        self, operator_idxs: LongTensorType
    ) -> Optional[FloatTensorType]:
        raise NotImplementedError("Regularizer not implemented for this operator")

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        return embs.abs()


DYNAMIC_OPERATORS = PluginRegistry[AbstractDynamicOperator]()


@DYNAMIC_OPERATORS.register_as("none")
class IdentityDynamicOperator(AbstractDynamicOperator):
    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return embeddings

    def get_operator_params_for_reg(
        self, operator_idxs: LongTensorType
    ) -> Optional[FloatTensorType]:
        return None


@DYNAMIC_OPERATORS.register_as("diagonal")
class DiagonalDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.diagonals = nn.Parameter(torch.ones((self.num_operations, self.dim)))

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return self.diagonals.to(device=embeddings.device)[operator_idxs] * embeddings

    def get_operator_params_for_reg(
        self, operator_idxs: LongTensorType
    ) -> Optional[FloatTensorType]:
        return self.diagonals.to(device=operator_idxs.device)[operator_idxs].abs()


@DYNAMIC_OPERATORS.register_as("translation")
class TranslationDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.translations = nn.Parameter(torch.zeros((self.num_operations, self.dim)))

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        return (
            embeddings + self.translations.to(device=embeddings.device)[operator_idxs]
        )

    def get_operator_params_for_reg(
        self, operator_idxs: LongTensorType
    ) -> Optional[FloatTensorType]:
        return self.translations.to(device=operator_idxs.device)[operator_idxs].abs()


@DYNAMIC_OPERATORS.register_as("linear")
class LinearDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(
            torch.diag_embed(torch.ones(()).expand(num_operations, dim))
        )

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        # We add a dimension so that matmul performs a matrix-vector product.
        return torch.matmul(
            self.linear_transformations.to(device=embeddings.device)[operator_idxs],
            embeddings.unsqueeze(-1),
        ).squeeze(-1)


@DYNAMIC_OPERATORS.register_as("affine")
class AffineDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        self.linear_transformations = nn.Parameter(
            torch.diag_embed(torch.ones(()).expand(num_operations, dim))
        )
        self.translations = nn.Parameter(torch.zeros((self.num_operations, self.dim)))

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        # We add a dimension so that matmul performs a matrix-vector product.
        return (
            torch.matmul(
                self.linear_transformations.to(device=embeddings.device)[operator_idxs],
                embeddings.unsqueeze(-1),
            ).squeeze(-1)
            + self.translations.to(device=embeddings.device)[operator_idxs]
        )

    # FIXME This adapts from the pre-D14024710 format; remove eventually.
    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        param_key = "%slinear_transformations" % prefix
        old_param_key = "%srotations" % prefix
        if old_param_key in state_dict:
            state_dict[param_key] = (
                state_dict.pop(old_param_key).transpose(-1, -2).contiguous()
            )
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


@DYNAMIC_OPERATORS.register_as("complex_diagonal")
class ComplexDiagonalDynamicOperator(AbstractDynamicOperator):
    def __init__(self, dim: int, num_operations: int):
        super().__init__(dim, num_operations)
        if dim % 2 != 0:
            raise ValueError(
                "Need even dimension as 1st half is real "
                "and 2nd half is imaginary coordinates"
            )
        self.real = nn.Parameter(torch.ones((self.num_operations, self.dim // 2)))
        self.imag = nn.Parameter(torch.zeros((self.num_operations, self.dim // 2)))

    def forward(
        self, embeddings: FloatTensorType, operator_idxs: LongTensorType
    ) -> FloatTensorType:
        match_shape(embeddings, ..., self.dim)
        match_shape(operator_idxs, *embeddings.size()[:-1])
        real_a = embeddings[..., : self.dim // 2]
        imag_a = embeddings[..., self.dim // 2 :]
        real_b = self.real.to(device=embeddings.device)[operator_idxs]
        imag_b = self.imag.to(device=embeddings.device)[operator_idxs]
        prod = torch.empty_like(embeddings)
        prod[..., : self.dim // 2] = real_a * real_b - imag_a * imag_b
        prod[..., self.dim // 2 :] = real_a * imag_b + imag_a * real_b
        return prod

    def get_operator_params_for_reg(self, operator_idxs) -> Optional[FloatTensorType]:
        return torch.sqrt(self.real[operator_idxs] ** 2 + self.imag[operator_idxs] ** 2)

    def prepare_embs_for_reg(self, embs: FloatTensorType) -> FloatTensorType:
        assert embs.shape[-1] == self.dim
        real, imag = embs[..., : self.dim // 2], embs[..., self.dim // 2 :]
        return torch.sqrt(real**2 + imag**2)


def instantiate_operator(
    operator: str, side: Side, num_dynamic_rels: int, dim: int
) -> Optional[Union[AbstractOperator, AbstractDynamicOperator]]:
    if num_dynamic_rels > 0:
        dynamic_operator_class = DYNAMIC_OPERATORS.get_class(operator)
        return dynamic_operator_class(dim, num_dynamic_rels)
    elif side is Side.LHS:
        return None
    else:
        operator_class = OPERATORS.get_class(operator)
        return operator_class(dim)
