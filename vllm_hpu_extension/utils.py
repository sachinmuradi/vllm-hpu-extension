###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import wraps
import os
from functools import lru_cache

import habana_frameworks.torch as htorch
import torch

from .cache_ops import insert_or_update_cache

@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'

def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = os.environ.get('VLLM_CONTIGUOUS_PA',
                                                'false').lower() == 'true'
        self.custom_pa = os.environ.get('VLLM_CUSTOM_PA',
                                                'false').lower() == 'true'

    def forward(self, input, cache, block_indices, block_offset, is_key_transposed=False):
        insert_or_update_cache(input, cache, block_indices, block_offset, is_key_transposed)
        return cache

    def fetch_from_cache(self, cache, blocks, is_key_transposed=False):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        elif is_key_transposed and not self.custom_pa:
            return cache.index_select(0, blocks).permute(0, 3, 2, 1).contiguous()
        else:
            return cache.index_select(0, blocks)

