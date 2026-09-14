import json
from pathlib import Path

from torch_llm.training.train_step import TrainStepOutput


class TrainingLogger:
    def __init__(self, path):
        self.path = path

    def log(self, model_metrics: TrainStepOutput, other_metrics: dict):
        overall_metrics = {}

        overall_metrics["Loss"] = model_metrics.loss.item()
        overall_metrics["LM Loss"] = model_metrics.lm_loss.item()
        overall_metrics["Aux Loss"] = model_metrics.aux_loss.item()

        if model_metrics.grad_norm is not None:
            overall_metrics["GradNorm"] = model_metrics.grad_norm.item()

        overall_metrics.update(other_metrics)

        Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        with open(self.path, "a") as f:
            f.write(json.dumps(overall_metrics) + "\n")

        for metric, stat in overall_metrics.items():
            print(f"{metric}: {stat}")
