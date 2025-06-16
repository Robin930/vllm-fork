from typing import Optional

import torch

from vllm.profiler.metrics.constants import OperationMetrics
from vllm.profiler.metrics.metrics_store import MetricsStore


class CudaTimer:

    def __init__(
        self,
        name: OperationMetrics,
        layer_id: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.name = name
        self.metrics_store = MetricsStore.get_instance()
        self.layer_id = layer_id
        self.disabled = (name is None) or not self.metrics_store.is_op_enabled(
            metric_name=self.name, layer_id=layer_id, rank=rank
        )
        self.use_cuda_events = False

        if self.disabled:
            return

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=self.handle_trace,
        )
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if self.disabled:
            return

        if self.use_cuda_events:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.profiler.__enter__()
        return self
    
    def handle_trace(self, trace):
        # print(trace.key_averages().table())
        # for e in trace.key_averages():
            # print(f'key: {e.key}, cuda_time: {e.cuda_time}, flops: {e.flops}, input_shapes: {e.input_shapes}')
        total_cuda_time = sum([e.device_time for e in trace.key_averages()])
        self.metrics_store.push_operation_metrics(
            self.name,
            total_cuda_time * 1e-3,
        )

    def __exit__(self, *args):
        if self.disabled:
            return

        if self.use_cuda_events:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()
            elapsed_time = self.start_event.elapsed_time(self.end_event)     # ms
            self.metrics_store.push_operation_metrics(
                self.name,
                elapsed_time
            )
        else:
            self.profiler.__exit__(*args)



