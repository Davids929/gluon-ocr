# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Building blocks and utility for models."""
__all__ = ['L2Normalization', 'GELU']

import math
from mxnet import ndarray
from mxnet.gluon import Block, HybridBlock

class L2Normalization(HybridBlock):
    """Normalize the input array by dividing the L2 norm along the given axis.

    ..code

        out = data / (sqrt(sum(data**2, axis)) + eps)

    Parameters
    ----------
    axis : int, default -1
        The axis to compute the norm value.
    eps : float, default 1E-6
        The epsilon value to avoid dividing zero
    """
    def __init__(self, axis=-1, eps=1E-6, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        ret = F.broadcast_div(x, F.norm(x, axis=self._axis, keepdims=True) + self._eps)
        return ret


class GELU(HybridBlock):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    https://arxiv.org/abs/1606.08415

    Parameters
    ----------
    approximate : bool, default False
        If True, use tanh approximation to calculate gelu. If False, use erf.

    """

    def __init__(self, approximate=False, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._approximate = approximate

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        Inputs:
            - **data**: input tensor with arbitrary shape.
        Outputs:
            - **out**: output tensor with the same shape as `data`.
        """
        if not self._approximate:
            return F.LeakyReLU(x, act_type='gelu')
        else:
            return 0.5 * x * (1 + F.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))))

    def __repr__(self):
        s = '{name}()'
        return s.format(name=self.__class__.__name__)
