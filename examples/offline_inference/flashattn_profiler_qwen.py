import os
import gc
import torch
import pandas as pd

from typing import List, Tuple, Any

from vllm.engine.llm_engine import EngineArgs
from vllm.config import (VllmConfig, set_current_vllm_config)
from vllm.attention import Attention
from vllm.attention.backends.flash_attn import (FlashAttentionImpl, FlashAttentionMetadata)
from vllm.profiler.metrics.metrics_store import MetricsStore
from vllm.profiler.metrics.constants import OperationMetrics
from vllm.profiler.cuda_timer import CudaTimer

from transformers import LlamaConfig
from tqdm import tqdm


os.environ['VLLM_USE_V1'] = '0'
os.environ['MY_CUDA_PROFILE'] = '1'

max_block_num = 16384
device = 'cuda:5'
dtype = torch.bfloat16
dtype_str = 'bfloat16'


class TestCaseGenerator:
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        max_tokens_in_batch: int,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.max_tokens_in_batch = max_tokens_in_batch

    def get_num_tokens_to_profile(
        self,
        max_num_tokens: int,
    ):
        NUM_TOKENS_SPACE = (
            list([1, 2, 4])
            + list(range(8, 1024, 8))
            + list(range(1024, 2 * 1024 + 1, 16))
            + list(range(2 * 1024, 4 * 1024 + 1, 32))
            + list(range(4 * 1024, 8 * 1024 + 1, 64))
            + list(range(8 * 1024, 16 * 1024 + 1, 128))
            + list(range(16 * 1024, 32 * 1024 + 1, 256))
            + list(range(32 * 1024, 64 * 1024 + 1, 512))
            + list(range(64 * 1024, 128 * 1024 + 1, 1024))
        )
        num_tokens_to_profile = []
        for num_tokens in NUM_TOKENS_SPACE:
            if num_tokens <= max_num_tokens:
                num_tokens_to_profile.append(num_tokens)
            else:
                break
        num_tokens_to_profile.sort(reverse=True)
        return num_tokens_to_profile

    # def generate_prefill_test_cases(self) -> List[Any]:
    #     test_cases = []
    #     for num_prefills in range(1, self.max_batch_size + 1):
    #         test_cases.extend(self._generate_prefill_test_cases_helper(num_prefills, self.max_tokens_in_batch, self.max_seq_len))
    #     return test_cases

    # TODO: chentong bs=1
    def generate_prefill_test_cases(self) -> List[List[int]]:
        test_cases = []
        for i in range(1, self.max_seq_len + 1, 1):
            test_cases.append([i])
        return test_cases
        
    def _generate_prefill_test_cases_helper(
        self,
        batch_size: int,
        budget: int,
        prev_num_tokens: int,       # 去重
    ) -> List[Any]:
        if batch_size == 0:
            return [[]]
        if budget == 0:
            # illegal test case
            return None
        tokens_to_profile = self.get_num_tokens_to_profile(min(budget, prev_num_tokens))
        test_cases = []
        for num_tokens in tokens_to_profile:
            sub_cases = self._generate_prefill_test_cases_helper(batch_size-1, budget-num_tokens, num_tokens)
            if sub_cases is not None:
                test_cases.extend([[num_tokens] + lst for lst in sub_cases])
        return test_cases

    def _construct_decode_case(self, context_len_sum: int) -> List[int]:
        case = []
        while context_len_sum > 0:
            if context_len_sum < self.max_seq_len:
                case.append(context_len_sum)
                context_len_sum = 0
            else:
                case.append(self.max_seq_len)
                context_len_sum -= self.max_seq_len
        return case

    # def _construct_decode_case(self, context_len_sum: int) -> List[List[int]]:
    #     cases = []
    #     for batch_size in range(1, self.max_batch_size + 1, 1):
    #         num_token1 = context_len_sum // batch_size
    #         if num_token1 > self.max_seq_len:
    #             continue
    #         num_token2 = context_len_sum - num_token1 * (batch_size - 1)
    #         case = [num_token1] * (batch_size - 1) + [num_token2]
    #         cases.append(case)
    #     return cases

    # enumerate all sum(context_len)
    def generate_decode_test_cases(self) -> List[List[int]]:
        test_cases = []
        for i in range(1, self.max_seq_len * self.max_batch_size + 1, 1):
            test_cases.append(self._construct_decode_case(i))
        return test_cases


class FlashAttnProfiler:
    def __init__(self, engine_args: EngineArgs):
        self.vllm_config: VllmConfig = engine_args.create_engine_config()
        self.cache_config = self.vllm_config.cache_config
        self.quant_config = self.vllm_config.quant_config
        self.model_config: LlamaConfig = self.vllm_config.model_config.hf_config
        self.num_heads = self.model_config.num_attention_heads
        self.num_kv_heads = self.model_config.num_key_value_heads
        self.hidden_size = self.model_config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.block_size = self.cache_config.block_size
        self.max_model_len = self.vllm_config.model_config.max_model_len
        self.metrics_store = MetricsStore.get_or_create_instance(0, self.vllm_config, False)
        self.prefill_timer = CudaTimer(OperationMetrics.ATTN_OP)
        self.decode_timer = CudaTimer(OperationMetrics.ATTN_OP)
        self.num_repeats = 3
        self.num_warm_up = 5
        self.dtype = dtype
        self.kv_cache_dtype = dtype
        self.test_case_generator = TestCaseGenerator(
            max_batch_size=128,
            max_seq_len=4096,
            max_tokens_in_batch=4096,
        )

        with set_current_vllm_config(self.vllm_config):
            self.init_device()

    def init_device(self):
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        torch.set_default_device(self.device)

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def init_kvcache(self, block_size: int, max_num_blocks: int, tp_size: int):
        self.kv_cache = torch.empty(
            (2, max_num_blocks, block_size, self.num_kv_heads // tp_size, self.head_dim),
            dtype=self.kv_cache_dtype,
            device=device,
        )

    def init_flash_attn(self, tp_size: int):
        self.attn = Attention(
            self.num_heads // tp_size,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads // tp_size,
            cache_config=self.cache_config,
            quant_config=self.quant_config,
            per_layer_sliding_window=False,
            prefix=f'flash_attn_tp{tp_size}',
        )
        self.attn_impl: FlashAttentionImpl = self.attn.impl

    def profile_prefill(
            self,
            num_prefills: int,
            prefill_tokens: List[int],
            tp_size: int,
        ):
        assert num_prefills == len(prefill_tokens)
        num_prefill_tokens = sum(prefill_tokens)
        seq_lens_tensor = torch.tensor(prefill_tokens, dtype=torch.int32, device=self.device)
        query_start_loc = torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device), torch.cumsum(seq_lens_tensor, dim=0)]).to(torch.int32)
        flash_attn_metadata = FlashAttentionMetadata(
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=torch.arange(1, num_prefill_tokens + 1, 1, dtype=torch.long, device=self.device),
            seq_lens=prefill_tokens,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=max(prefill_tokens),
            max_decode_seq_len=0,
            context_lens_tensor=torch.zeros((num_prefills,), dtype=torch.int32, device=self.device),
            block_tables=torch.zeros((num_prefills, 0), dtype=torch.int32, device=self.device),
            use_cuda_graph=False,
            max_query_len=max(prefill_tokens),
            max_decode_query_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=query_start_loc,
            multi_modal_placeholder_index_maps={},
            enable_kv_scales_calculation=True,
        )
        query = torch.randn(
            (num_prefill_tokens, self.num_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        key = torch.randn(
            (num_prefill_tokens, self.num_kv_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        value = torch.randn(
            (num_prefill_tokens, self.num_kv_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.zeros(
            (num_prefill_tokens, self.num_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        with self.prefill_timer:
            self.attn_impl.forward(
                self.attn,
                query,
                key,
                value,
                self.kv_cache,
                flash_attn_metadata,
                output
            )
    
    def get_block_tables(
        self,
        num_decodes: int,
        seq_lens: List[int],
    ) -> Tuple[torch.Tensor, bool]:
        assert len(seq_lens) == num_decodes
        num_blocks = [(l + self.block_size - 1) // self.block_size for l in seq_lens]
        max_num_blocks = max(num_blocks)
        block_tables = torch.zeros((num_decodes, max_num_blocks), dtype=torch.int32, device=self.device)
        
        current_block_id = 1
        for i, count in enumerate(num_blocks):
            block_tables[i, :count] = torch.arange(current_block_id, current_block_id + count, dtype=torch.int32, device=self.device)
            current_block_id += count
        return block_tables, current_block_id >= max_block_num
        

    """
    多了一个这个(add), 不过影响不是很大
    key: void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<int> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<int> const&)::{lambda(int)#1}), cuda_time: 2.399999999999636, flops: 0, input_shapes: 
    """
    def profile_decode(
            self,
            num_decodes: int,
            context_lens: List[int],
            tp_size: int,
        ):
        seq_lens = [x + 1 for x in context_lens]
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        cu_seq_lens_tensor = torch.cat([torch.tensor([0], dtype=torch.int32, device=self.device), torch.cumsum(seq_lens_tensor, dim=0)]).to(torch.int32)
        block_tables, overflow = self.get_block_tables(num_decodes, seq_lens)
        if overflow:
            return
        flash_attn_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=num_decodes,
            slot_mapping=torch.arange(1, num_decodes + 1, 1, dtype=torch.long, device=self.device),
            seq_lens=seq_lens,
            seq_lens_tensor=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            max_prefill_seq_len=0,
            max_decode_seq_len=max(seq_lens),
            context_lens_tensor=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=block_tables,
            use_cuda_graph=False,
            max_query_len=1,
            max_decode_query_len=1,
            query_start_loc=torch.arange(num_decodes + 1, dtype=torch.int32, device=self.device),
            seq_start_loc=cu_seq_lens_tensor,
            multi_modal_placeholder_index_maps={},
            enable_kv_scales_calculation=True,
        )
        query = torch.randn(
            (num_decodes, self.num_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        key = torch.randn(
            (num_decodes, self.num_kv_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        value = torch.randn(
            (num_decodes, self.num_kv_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.zeros(
            (num_decodes, self.num_heads // tp_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        with self.decode_timer:
            self.attn_impl.forward(
                self.attn,
                query,
                key,
                value,
                self.kv_cache,
                flash_attn_metadata,
                output
            )

    def warm_up(self, tp_size):
        for _ in range(self.num_warm_up):
            self.profile_prefill(1, [1024], tp_size)
        for _ in range(self.num_warm_up):
            self.profile_decode(8, [64, 64, 64, 64, 64, 64, 64, 64], tp_size)
        self.metrics_store.reset()

    def profile_tp(self, tp_size: int, prefill_test_cases, decode_test_cases, pbar):
        self.init_kvcache(self.block_size, max_block_num, tp_size)
        self.init_flash_attn(tp_size)
        self.warm_up(tp_size)
        all_results = []
        # prefill_test_cases = self.test_case_generator.generate_prefill_test_cases()
        # decode_test_cases = self.test_case_generator.generate_decode_test_cases()
        # pbar = tqdm(total=(len(prefill_test_cases) + len(decode_test_cases)))
        # prefill
        for prefill_tokens in prefill_test_cases:
            num_prefills = len(prefill_tokens)
            self.metrics_store.reset()
            for _ in range(self.num_repeats):
                self.profile_prefill(num_prefills, prefill_tokens, tp_size)
            time_stats = self.metrics_store.get_stats()
            stats = {
                'time_stats': time_stats,
                'n_embd': self.hidden_size,
                'n_q_head': self.num_heads,
                'n_kv_head': self.num_kv_heads,
                'block_size': self.block_size,
                'num_tensor_parallel_workers': tp_size,       
                'max_model_len': self.max_model_len,
                'batch_size': 1,    # 1 for prefill
                'prefill_chunk_size': sum(prefill_tokens),       # used as prefill tokens in total
                'kv_cache_size': 0,
                'is_prefill': 1,
                'attention_backend': 'flashattention3',
            }
            all_results.append(stats)
            pbar.update(1)

        # decode
        for context_lens in decode_test_cases:
            num_decodes = len(context_lens)
            self.metrics_store.reset()
            for _ in range(self.num_repeats):
                self.profile_decode(num_decodes, context_lens, tp_size)
            time_stats = self.metrics_store.get_stats()
            pbar.update(1)
            if len(time_stats) == 0:
                continue
            stats = {
                'time_stats': time_stats,
                'n_embd': self.hidden_size,
                'n_q_head': self.num_heads,
                'n_kv_head': self.num_kv_heads,
                'block_size': self.block_size,
                'num_tensor_parallel_workers': tp_size,       
                'max_model_len': self.max_model_len,
                'batch_size': num_decodes,    
                'prefill_chunk_size': 0,       # not used
                'kv_cache_size': sum(context_lens),
                'is_prefill': 0,
                'attention_backend': 'flashattention3',
            }
            all_results.append(stats)
        return all_results

    def profile(self):
        # tp_sizes_to_profile = [1, 2, 4, 8]
        tp_sizes_to_profile = [2]
        prefill_test_cases = self.test_case_generator.generate_prefill_test_cases()
        decode_test_cases = self.test_case_generator.generate_decode_test_cases()
        pbar = tqdm(total=(len(prefill_test_cases) + len(decode_test_cases)) * len(tp_sizes_to_profile))
        all_results = []
        for tp_size in tp_sizes_to_profile:
            all_results.extend(self.profile_tp(tp_size, prefill_test_cases, decode_test_cases, pbar))

        df = pd.DataFrame(all_results)
        df = (
            pd.json_normalize(df['time_stats'])
            .add_prefix('time_stats.')
            .join(df.drop(columns=['time_stats']))
        )
        return df


def main():
    model_path = '/model/Qwen2.5-32B'
    model_name = 'Qwen2.5-32B'
    engine_args = EngineArgs(
        model= model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype=dtype_str,
    )
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    profiler = FlashAttnProfiler(engine_args)
    with set_current_vllm_config(profiler.vllm_config):
        # profiler.init_kvcache(profiler.block_size, 16384, 1)
        # profiler.init_flash_attn(1)
        # profiler.warm_up(1)
        # profiler.metrics_store.reset()
        # profiler.profile_decode(32, [4096] * 32, 1)
        # print(profiler.metrics_store.operation_metrics)
        df = profiler.profile()
        os.makedirs(f'./profile/{model_name}', exist_ok=True)
        df.to_csv(f'./profile/{model_name}/attention.csv', index=False)


if __name__ == '__main__':
    main()