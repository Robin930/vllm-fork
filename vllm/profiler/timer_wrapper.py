import time
from typing import Optional
from functools import wraps

from vllm.profiler.metrics.constants import OperationMetrics
from vllm.profiler.metrics.metrics_store import MetricsStore

class ExecutionTimer:
    def __init__(
        self,
        func = None,
        *,
        name: OperationMetrics = None,
        layer_id: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.func = func
        self.name = name
        self.metrics_store = MetricsStore.get_instance()
        self.layer_id = layer_id
        self.disabled = (name is None) or not self.metrics_store.is_op_enabled(
            metric_name=self.name, layer_id=layer_id, rank=rank
        )

        if self.disabled:
            return

        self.start_time = None
        self.end_time = None

    def __call__(self, *args, **kwargs):
        if self.disabled:
            return self.func(*args, **kwargs)

        self.start_time = time.time_ns()
        result = self.func(*args, **kwargs)
        self.end_time = time.time_ns()
        elapsed_time_ns = self.end_time - self.start_time
        self.metrics_store.push_operation_metrics(
            self.name,
            elapsed_time_ns * 1e-6,  # convert to ms
        )
        return result

    def __enter__(self):
        if self.disabled:
            return self

        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.disabled:
            return
        
        self.end_time = time.time_ns()
        elapsed_time_ns = self.end_time - self.start_time
        self.metrics_store.push_operation_metrics(
            self.name,
            elapsed_time_ns * 1e-6,  # convert to ms
        )
