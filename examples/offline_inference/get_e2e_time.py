import os
import time
import argparse
import torch

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine
from vllm.profiler.metrics.metrics_store import MetricsStore
from tqdm import tqdm

os.environ['VLLM_USE_V1'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--prompt_name', type=str, default='summary')
parser.add_argument('--resp_name', type=str, default='document')
parser.add_argument('--local_dataset', action='store_true', default=True)
parser.add_argument('--model', type=str)
parser.add_argument('--tp', type=int, default=1)

args = parser.parse_args()
dataset = args.dataset
dataset_name = dataset.split('/')[-1]
local_dataset = args.local_dataset
model = args.model
model_name = model.split('/')[-1]
prompt_name = args.prompt_name
resp_name = args.resp_name
tp = args.tp

if local_dataset:
    ds = load_from_disk(dataset)
else:
    ds = load_dataset(dataset, split='train')

tokenizer = AutoTokenizer.from_pretrained(model)

llm = LLM(
    model=model,
    tensor_parallel_size=tp,
    enforce_eager=False,
    max_model_len=4096,
    block_size=16,
    max_num_batched_tokens=4096,
    max_num_seqs=128,
)
engine: LLMEngine = llm.llm_engine
metrics_store: MetricsStore = engine.metrics_store

total_req = len(ds)
pbar = tqdm(total=total_req, desc='add_request')
for i, req in enumerate(ds):
    prompt = req[prompt_name]
    resp = req[resp_name]
    output_tokens = tokenizer(resp, truncation=True)
    output_len = len(output_tokens['input_ids'])
    params = SamplingParams(
        ignore_eos=True,
        max_tokens=output_len, 
    )
    engine.add_request(f'req{i}', prompt, params)
    pbar.update(1)

pbar = tqdm(total=total_req, desc='exec')
metrics_store.reset()
unfinished_req = total_req
while engine.has_unfinished_requests():
    start = time.time()
    engine.step()
    end = time.time()
    time_ms = (end - start) * 1000
    metrics_store.push_e2e_metrics(time_ms)
    if engine.get_num_unfinished_requests() < unfinished_req:
        pbar.update(unfinished_req - engine.get_num_unfinished_requests())
        unfinished_req = engine.get_num_unfinished_requests()

metrics_base_dir = f'./e2e_time/{model_name}/{dataset_name}'
os.makedirs(metrics_base_dir, exist_ok=True)
metrics_store._store_e2e_metrics(metrics_base_dir)
stats = metrics_store.get_e2e_stats()
print(stats)