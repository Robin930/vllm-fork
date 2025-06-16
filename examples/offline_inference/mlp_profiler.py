# monkey mock llama
from mocked_llama import LlamaModel
from mocked_qwen2 import Qwen2Model
from transformers import LlamaConfig
import vllm.distributed

from vllm.engine.llm_engine import EngineArgs
from vllm.config import (VllmConfig, set_current_vllm_config)
from vllm.profiler.metrics.metrics_store import MetricsStore
import vllm

import pandas as pd
import torch
import gc
import os
from typing import List
from tqdm import tqdm

os.environ['VLLM_USE_V1'] = '0'
os.environ['MY_CUDA_PROFILE'] = '1'

# 1044

def get_num_tokens_to_profile(
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

class MLPProfiler:
    def __init__(
        self,
        engine_args: EngineArgs,
        model_type: str = 'llama',
    ):
        self.model_type = model_type
        self.vllm_config: VllmConfig = engine_args.create_engine_config()
        self.vllm_config.parallel_config.tensor_parallel_size = 1
        self.vllm_config.parallel_config.pipeline_parallel_size = 1
        self.model_config = self.vllm_config.model_config
        self.llama_config: LlamaConfig = self.model_config.hf_config
        self.warm_up_steps = 10
        self.num_repeats = 3

        self.dtype = self.model_config.dtype
        if self.model_config.max_model_len > 4096:
            self.model_config.max_model_len = 4096
        self.max_model_len = self.model_config.max_model_len
        self.hidden_size = self.model_config.get_hidden_size()
        self.vocab_size = self.model_config.get_vocab_size()
        self.num_heads = self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)    # need set tp = 1
        self.num_kv_heads = self.model_config.get_total_num_kv_heads()
        self.head_dim = self.model_config.get_head_size()  
        self.num_layers = self.model_config.get_num_layers(self.vllm_config.parallel_config)

        torch.set_default_dtype(self.dtype)
        

        with set_current_vllm_config(self.vllm_config):
            self.init_device()
            self.metrics_store = MetricsStore.get_or_create_instance(0, self.vllm_config, True)

    def init_device(self) -> None:
        if self.vllm_config.device_config.device.type == 'cuda':
            self.device = torch.device(f'cuda:0')
            torch.cuda.set_device(self.device)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        else:
            raise NotImplementedError

        vllm.distributed.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method='tcp://127.0.0.1:12350',
            backend='nccl',
        )
        vllm.distributed.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend='nccl',
        )

    def prepare_testbench(self):
        num_tokens_to_profile = get_num_tokens_to_profile(self.max_model_len)
        return num_tokens_to_profile

    def warm_up(self):
        for i in range(self.warm_up_steps):
            self.execute_model(2000)
        self.metrics_store.reset()
        print('warm up end')

    def execute_model(self, num_tokens: int):
        with set_current_vllm_config(self.vllm_config):
            input_tokens = torch.randint(
                low=0,
                high=self.model_config.get_vocab_size(),
                size=(num_tokens,),
                device='cuda:0',
                dtype=torch.long,
            )
            positions = torch.arange(num_tokens, device='cuda:0', dtype=torch.long)
            self.model.forward(input_tokens, positions)

    def profile_tp(self, tp_size: int, num_tokens_to_profile: List[int], pbar):
        print(f'profiling tp_size: {tp_size}')
        if self.model_type == 'llama':
            self.model = LlamaModel(vllm_config=self.vllm_config, mocked_tp_size=tp_size)
        elif self.model_type == 'qwen2':
            self.model = Qwen2Model(vllm_config=self.vllm_config, mocked_tp_size=tp_size)
        else:
            raise NotImplementedError
        self.model = self.model.to('cuda:0')
        all_results = []
        self.warm_up()
        for num_tokens in num_tokens_to_profile:
            self.metrics_store.reset()
            for _ in range(self.num_repeats):
                self.execute_model(num_tokens)
            time_stats = self.metrics_store.get_stats()
            stats = {
                'time_stats': time_stats,
                'n_head': self.num_heads,
                'n_kv_head': self.num_kv_heads,
                'n_embd': self.hidden_size,
                'n_expanded_embd': 29568,   # TODO
                # 'n_expanded_embd': self.llama_config.intermediate_size,
                'vocab_size': self.vocab_size,
                'num_tokens': num_tokens,
                'use_gated_mlp': True,      # TODO: chentong what is this?
                'num_tensor_parallel_workers': tp_size,       
            }
            all_results.append(stats)
            pbar.update(1)
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        return all_results

    def profile(self):
        all_results = []
        # tp_sizes_tp_profile = [1, 2, 4, 8]
        tp_sizes_tp_profile = [4]
        num_tokens_to_profile = self.prepare_testbench()
        pbar = tqdm(total=len(tp_sizes_tp_profile)*len(num_tokens_to_profile))
        for tp_size in tp_sizes_tp_profile:
            all_results.extend(self.profile_tp(tp_size, num_tokens_to_profile, pbar))
            
        df = pd.DataFrame(all_results)
        # print(df.to_string(index=False))
        df = (
            pd.json_normalize(df['time_stats'])
            .add_prefix('time_stats.')
            .join(df.drop(columns=['time_stats']))
        )
        return df

def main():
    model_path = '/model/Qwen2.5-72B-Instruct'
    model_name = 'Qwen2.5-72B-Instruct'
    engine_args = EngineArgs(
        # model='/workspace/Llama-2-7b-hf',
        model=model_path,
        enforce_eager=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    torch.set_default_device('cuda:0')
    mlp_profiler = MLPProfiler(engine_args, model_type='qwen2')
    with set_current_vllm_config(mlp_profiler.vllm_config):
        df = mlp_profiler.profile()
    os.makedirs(f'./profile/{model_name}', exist_ok=True)
    df.to_csv(f'./profile/{model_name}/mlp.csv', index=False)


if __name__ == '__main__':
    main()