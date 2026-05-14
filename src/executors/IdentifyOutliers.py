"""
    Detects outlier embeddings using von Mises-Fisher directional statistics.
    Injects the outlier analysis results into the incoming Detection objects under the 'Identify' key,
    and removes the heavy SIFT 'keyPoints' payload.
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

        self.detections = self.request.get_param("inputDetections")
        self.threshold_percentile = float(self.request.get_param("configThresholdPercentile") or 0.05)
        self.warmup = int(self.request.get_param("configWarmup") or 10)
        self.window_size = int(self.request.get_param("configWindowSize") or 32)

        self.sample_count = self.bootstrap.get("sample_count", 0)
        self.embedding_window = self.bootstrap.get("embedding_window", [])

        self.is_outlier = False
        self.percentile = 0.5
        self.warming_up = True

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {
            "sample_count": 0,
            "embedding_window": [],
        }

    def _extract_embedding(self):
        detections = self.detections
        if detections is None:
            return None

        if not isinstance(detections, list):
            detections = [detections]

        all_descriptors = []
        for det in detections:
            key_points = det.get('keyPoints') if isinstance(det, dict) else getattr(det, 'keyPoints', None)
            
            if key_points is None:
                continue
                
            for kp in key_points:
                descriptor = kp.get('descriptor') if isinstance(kp, dict) else getattr(kp, 'descriptor', None)
                
                if descriptor is not None:
                    desc = np.array(descriptor, dtype=np.float32)
                    if desc.shape[0] > 0:
                        all_descriptors.append(desc)

        if not all_descriptors:
            return None

        return np.mean(all_descriptors, axis=0)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
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
        embedding = self._extract_embedding()

        if embedding is None:
            self.warming_up = self.sample_count < self.warmup
        else:
            normalized = self._normalize(embedding)
            self.sample_count += 1
            self.bootstrap["sample_count"] = self.sample_count
            self.warming_up = self.sample_count <= self.warmup

            self.embedding_window.append(normalized.tolist())
            if len(self.embedding_window) > self.window_size:
                self.embedding_window = self.embedding_window[-self.window_size:]
            self.bootstrap["embedding_window"] = self.embedding_window

            if not self.warming_up:
                window_arr = np.array(self.embedding_window, dtype=np.float32)
                mu = self._normalize(np.mean(window_arr, axis=0))
                self.percentile = self._compute_percentile(normalized, mu)
                self.is_outlier = (
                    self.percentile < self.threshold_percentile
                    or self.percentile > (1.0 - self.threshold_percentile)
                )

        identify_result = {
            "is_outlier": bool(self.is_outlier),
            "percentile": round(float(self.percentile), 4),
            "warming_up": bool(self.warming_up)
        }

        enriched_detections = []
        if self.detections:
            det_list = self.detections if isinstance(self.detections, list) else [self.detections]
            for det in det_list:
                det_dict = det if isinstance(det, dict) else getattr(det, "model_dump", lambda: getattr(det, "dict", lambda: vars(det))())()
                
                # Devasa SIFT verisini çıktıdan temizliyoruz
                det_dict.pop("keyPoints", None)
                
                det_dict["Identify"] = identify_result
                enriched_detections.append(det_dict)
                
        self.detections = enriched_detections

    def run(self):
        self.detect()
        package_model = build_response(context=self)
        return package_model

if "__main__" == __name__:
    Executor(sys.argv[1]).run()