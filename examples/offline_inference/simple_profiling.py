# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch

from vllm import LLM, SamplingParams
from vllm.engine.llm_engine import LLMEngine

enable_vllm_profile = True
enable_my_cuda_profile = False
enable_my_cpu_profile = False

os.environ["VLLM_USE_V1"] = "0"
if enable_vllm_profile:
    os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

if enable_my_cuda_profile:
    os.environ["MY_CUDA_PROFILE"] = "1"

if enable_my_cpu_profile:
    os.environ["MY_CPU_PROFILE"] = "1"
    

# enable torch profiler, can also be set on cmd line
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Sample prompts.
prompts = [
    # "Hello, my name is",
    "The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is The president of the United States is ",
    # "The capital of France is",
    # "The future of AI is",
]

prompts2 = [
    "Hello, my name is",
    "The capital of France is The capital of France is of France is ",
    "The capital of France is The capital of France is The of France is ",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, ignore_eos=True, max_tokens=20)

if __name__ == "__main__":

    enforce_eager = False
    print(f'enforce eager: {enforce_eager}')

    # Create an LLM.
    # llm = LLM(model="/workspace/Llama-2-7b-hf", tensor_parallel_size=2, enforce_eager=enforce_eager)
    llm = LLM(model="/model/Llama-2-70b-hf", tensor_parallel_size=2, enforce_eager=enforce_eager)

    # warm up
    llm.generate(prompts, sampling_params)
    print('warm up end')

    if enable_vllm_profile:
        llm.start_profile()
    else:
        engine: LLMEngine = llm.llm_engine
        engine.reset_metrics_store()

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    if enable_vllm_profile:
        llm.stop_profile()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, tokens: {len(output.prompt_token_ids)}, Generated text: {generated_text!r}, tokens: {len(output.outputs[0].token_ids)}")
        print(f"TTFT: {output.metrics.first_token_time - output.metrics.arrival_time}")

    if enable_my_cpu_profile or enable_my_cuda_profile:
        engine.dump_metrics_store()

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)
