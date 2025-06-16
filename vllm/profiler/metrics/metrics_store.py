from typing import Any, Dict, List, Union, Tuple, Optional
import os
import numpy as np

from vllm.config import VllmConfig
from vllm.profiler.metrics.constants import (
    BatchMetricsCountDistribution,
    BatchMetricsTimeDistribution,
    CompletionMetricsTimeSeries,
    CpuOperationMetrics,
    OperationMetrics,
    SequenceMetricsHistogram,
    SequenceMetricsTimeDistributions,
    TokenMetricsTimeDistribution,
    TokenMetricsTimeList,
)
# from vllm.profiler.metrics.cdf_sketch import CDFSketch
# from vllm.profiler.metrics.data_series import DataSeries
# from vllm.outputs import RequestOutput
# from vllm.sequence import SequenceGroup


import torch


def check_enabled(func):

    def wrapper(self, *args, **kwargs):
        if self.disabled:
            return
        return func(self, *args, **kwargs)

    return wrapper

PROFILE_LAYER_ID = 10
BATCH_ID_STR = "Batch Id"
REQUEST_ID_STR = "Request Id"
DECODE_TOKEN_ID_STR = "Decode Token Id"
COUNT_STR = "Count"
TIME_STR = "Time (sec)"
TIME_STR_MS = "Time (ms)"
OPERATION_STR = "Operation"

class MetricsStore:
    _instance = None

    def __init__(
        self,
        rank: Optional[int],
        vllm_config: VllmConfig,
        is_global: bool,
    ):
        
        # TODO: chentong add config
        self.disabled = os.environ.get('MY_CUDA_PROFILE') is None and os.environ.get('MY_CPU_PROFILE') is None
        self.cuda_disabled = os.environ.get('MY_CUDA_PROFILE') is None

        self.rank = rank
        self.is_global = is_global
        self.output_dir = './vllm_metrics'
        self.is_prefill = True

        self.reset()

    def set_prefill(self, is_prefill: bool):
        self.is_prefill = is_prefill

    def is_op_enabled(
        self,
        metric_name: Any,
        rank: Optional[int] = None,
        layer_id: Optional[int] = None,
    ) -> bool:
        if self.disabled:
            return False
        if self.cuda_disabled and metric_name in OperationMetrics:
            return False
        return True
        

    @classmethod
    def get_or_create_instance(
        cls,
        rank: Optional[int],
        vllm_config: VllmConfig,
        is_global: bool,
    ):
        if cls._instance is None:
            cls._instance = cls(rank, vllm_config, is_global)
        if rank is not None:
            cls._instance.rank = rank
        cls._instance.is_global = cls._instance.is_global or is_global
        return cls._instance
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            raise RuntimeError('metrics_store not initialized')
        return cls._instance
        
    def reset(self):
        # if self.disabled:
        #     return

        self.operation_metrics: Dict[OperationMetrics, List[float]] = {}
        self.cpu_operation_metrics: Dict[CpuOperationMetrics, List[float]] = {}
        self.e2e_metrics: Dict[SequenceMetricsTimeDistributions, List[float]] = {}
        self.e2e_metrics[SequenceMetricsTimeDistributions.TTFT] = []
        self.e2e_metrics[SequenceMetricsTimeDistributions.TBT] = []


    @check_enabled
    def on_request_arrival(
        self,
    ):
        raise NotImplementedError

    @check_enabled
    def on_schedule(
        self,
    ):
        raise NotImplementedError
    
    # 一个iteration结束
    @check_enabled
    def on_batch_end(
        self,
    ):
        raise NotImplementedError
        
    @check_enabled
    def push_operation_metrics(
        self,
        metrics_name: OperationMetrics,
        time: float, # in ms
    ):
        # if metrics_name == OperationMetrics.ATTN_DECODE:
        #     print(time * 1000)
        
        # print(f'{metrics_name.name.lower()}: {time}ms')
        if metrics_name not in self.operation_metrics:
            self.operation_metrics[metrics_name] = []
        self.operation_metrics[metrics_name].append(time)

    @check_enabled
    def push_cpu_operation_metrics(
        self,
        metrics_name: CpuOperationMetrics,
        time: float, # in ms
    ):
        # print(f'{metrics_name.name.lower()}: {time}ms')
        if metrics_name not in self.cpu_operation_metrics:
            self.cpu_operation_metrics[metrics_name] = []
        self.cpu_operation_metrics[metrics_name].append(time)

    def push_e2e_metrics(
        self,
        time: float, # in ms
    ):
        if self.is_prefill:
            self.e2e_metrics[SequenceMetricsTimeDistributions.TTFT].append(time)
        else:
            self.e2e_metrics[SequenceMetricsTimeDistributions.TBT].append(time)

    @check_enabled
    def dump(self, is_global):
        if is_global:
            self._dump_global()
        else:
            self._dump_local()

    def _dump_global(self):
        base_path = f'{self.output_dir}/global'
        os.makedirs(base_path, exist_ok=True)

    
    def _dump_local(self):
        base_path = f'{self.output_dir}/rank_{self.rank}'
        os.makedirs(base_path, exist_ok=True)
        self._store_operation_metrics(base_path)
        
    @check_enabled
    def _store_operation_metrics(self, base_path: str):
        for metric_name, time_lst in self.operation_metrics.items():
            file = f'{base_path}/{metric_name.value}.csv'
            print(f'dump {file}')
            with open(file, 'w') as f:
                for i, time in enumerate(time_lst):
                    f.write(f'{i},{metric_name.value},{time}\n')

    def _store_e2e_metrics(self, base_path: str):
        for metric_name, time_lst in self.e2e_metrics.items():
            file = f'{base_path}/{metric_name.name.lower()}.csv'
            print(f'dump {file}')
            with open(file, 'w') as f:
                for i, time in enumerate(time_lst):
                    f.write(f'{i},{metric_name.name.lower()},{time}\n')

    def get_stats(self):
        stats = {}
        for name, times in self.operation_metrics.items():
            stats[name.name.lower()] = {
                'total': len(times),
                'min': np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
        return stats

    def get_e2e_stats(self):
        stats = {}
        for name, times in self.e2e_metrics.items():
            stats[name.name.lower()] = {
                # 'total': len(times),
                'min': np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
        return stats

    def get_cpu_stats(self):
        stats = {}
        for name, times in self.cpu_operation_metrics.items():
            stats[name.name.lower()] = {
                'min': np.min(times),
                "max": np.max(times),
                "mean": np.mean(times),
                "median": np.median(times),
                "std": np.std(times),
            }
        return stats