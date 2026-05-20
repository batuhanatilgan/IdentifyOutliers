"""
    Detects outlier embeddings using von Mises-Fisher directional statistics.
    Reads pre-calculated embeddings (e.g., from CLIP) and injects the 'Identify'
    anomaly analysis results into the output payload.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.IdentifyOutliers.src.utils.response import build_response
from components.IdentifyOutliers.src.models.PackageModel import PackageModel


class IdentifyOutliers(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.input_data = self.request.get_param("inputData")
        self.threshold_percentile = float(self.request.get_param("configThresholdPercentile") or 0.05)
        self.warmup = int(self.request.get_param("configWarmup") or 10)
        self.window_size = int(self.request.get_param("configWindowSize") or 32)

        self.sample_count = self.bootstrap.get("sample_count", 0)
        self.embedding_window = self.bootstrap.get("embedding_window", [])
        
        self.output_data = []

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "sample_count": 0,
            "embedding_window": [],
        }

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    def _compute_percentile(self, current_normalized: np.ndarray, mu: np.ndarray) -> float:
        window = np.array(self.embedding_window, dtype=np.float32)
        historical_scores = window @ mu
        current_score = float(np.dot(current_normalized, mu))
        n = len(historical_scores)
        rank = np.sum(historical_scores < current_score)
        return float(rank) / float(n) if n > 0 else 0.5

    def detect(self):
        if not self.input_data:
            self.output_data = []
            return

        data_list = self.input_data if isinstance(self.input_data, list) else [self.input_data]
        enriched_data = []

        for item in data_list:
            item_dict = item if isinstance(item, dict) else getattr(item, "model_dump", lambda: getattr(item, "dict", lambda: vars(item))())()
            embedding = item_dict.get("embedding")
            
            is_outlier = False
            percentile = 0.5
            warming_up = True

            if embedding is not None:
                emb_arr = np.array(embedding, dtype=np.float32)
                normalized = self._normalize(emb_arr)
                
                self.sample_count += 1
                self.bootstrap["sample_count"] = self.sample_count
                warming_up = self.sample_count <= self.warmup

                self.embedding_window.append(normalized.tolist())
                if len(self.embedding_window) > self.window_size:
                    self.embedding_window = self.embedding_window[-self.window_size:]
                self.bootstrap["embedding_window"] = self.embedding_window

                if not warming_up:
                    window_arr = np.array(self.embedding_window, dtype=np.float32)
                    mu = self._normalize(np.mean(window_arr, axis=0))
                    percentile = self._compute_percentile(normalized, mu)
                    is_outlier = (percentile < self.threshold_percentile or percentile > (1.0 - self.threshold_percentile))
            item_dict.pop("embedding", None)
            item_dict["Identify"] = {
                "is_outlier": bool(is_outlier),
                "percentile": round(float(percentile), 4),
                "warming_up": bool(warming_up)
            }
            
            enriched_data.append(item_dict)

        self.output_data = enriched_data

    def run(self):
        self.detect()
        package_model = build_response(context=self)
        return package_model

if "__main__" == __name__:
    Executor(sys.argv[1]).run()