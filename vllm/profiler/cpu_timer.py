import time

from vllm.profiler.metrics.constants import CpuOperationMetrics
from vllm.profiler.metrics.metrics_store import MetricsStore


class CPUTimer:
    def __init__(
        self,
        name: CpuOperationMetrics,
    ):
        self.name = name
        self.metrics_store = MetricsStore.get_instance()
        self.disabled = (name is None) or not self.metrics_store.is_op_enabled(
            metric_name=name
        )

        if self.disabled:
            return

        self.start = None
        self.end = None

    def __enter__(self):
        if self.disabled:
            return
        
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if self.disabled:
            return

        self.end = time.time()
        elapsed_time = (self.end - self.start) * 1e3       # s -> ms
        self.metrics_store.push_cpu_operation_metrics(self.name, elapsed_time)

    # def timer(self, func):
    #     def wrapper(*args, **kwargs):
    #         self.start = time.time()
    #         result = func(*args, **kwargs)
    #         self.end = time.time()
    #         elapsed_time = (self.end - self.start) * 1e3
    #         self.metrics_store.push_cpu_operation_metrics(self.name, elapsed_time)
    #         return result
    #     return wrapper