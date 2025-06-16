import os
import pandas as pd
import numpy as np

from vllm import LLM, SamplingParams
from vllm.profiler.metrics.metrics_store import MetricsStore
from vllm.engine.llm_engine import LLMEngine

from tqdm import tqdm

os.environ['VLLM_USE_V1'] = '0'

NUM_PREFILL_TOKEN = 256
NUM_DECODE_TOKEN_AMPLIFICATION_FACTOR = 3

class CPUProfiler:
    def __init__(self):
        self.llm = LLM(model='/workspace/Llama-2-7b-hf')
        self.engine = self.llm.llm_engine
        self.metrics_store = MetricsStore.get_instance()
        self.max_batch_size = 128
        self.num_warm_up = 5
        self.num_repeats = 3

    def _get_input_params(self, batch_size: int) -> SamplingParams:
        sampling_params = SamplingParams(
            ignore_eos=True,
            max_tokens=batch_size * NUM_DECODE_TOKEN_AMPLIFICATION_FACTOR,
        )
        prompt_token_ids = (
            np.random.default_rng()
            .integers(low=0, high=10000, size=NUM_PREFILL_TOKEN)
        )

        return {
            'prompt': {'prompt_token_ids': prompt_token_ids},
            'params': sampling_params,
        }
        

    def warm_up(self):
        self.engine.add_request(request_id=f'req_0', **self._get_input_params(1))

        while self.engine.has_unfinished_requests():
            self.engine.step()
        self.metrics_store.reset()

    def profile_batch(self, batch_size: int):
        for i in range(batch_size):
            self.engine.add_request(request_id=f'req_{i}', **self._get_input_params(batch_size))

        while self.engine.has_unfinished_requests():
            self.engine.step()

    def profile(self):
        all_results = []
        self.warm_up()
        pbar = tqdm(total=self.max_batch_size)
        for batch_size in range(1, self.max_batch_size + 1, 1):
            self.metrics_store.reset()
            for _ in range(self.num_repeats):
                self.profile_batch(batch_size)
            cpu_time_stats = self.metrics_store.get_cpu_stats()
            stats = {
                'time_stats': cpu_time_stats,
                'batch_size': batch_size,
                'model_name': 'NousResearch/Llama-2-7b-hf',
                'tensor_parallel_degree': 1,
            }
            all_results.append(stats)
            pbar.update(1)

        df = pd.DataFrame(all_results)
        df = (
            pd.json_normalize(df['time_stats'])
            .join(df.drop(columns=['time_stats']))
        )
        return df

def main():
    profiler = CPUProfiler()
    df = profiler.profile()
    os.makedirs(f'./profile/llama2-7b', exist_ok=True)
    df.to_csv(f'./profile/llama2-7b/cpu_overheads.csv', index=False)
        
        
if __name__ == '__main__':
    main()