from typing import List, Optional
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    name: str = Field(default="benchmark_run")
    seed: int = Field(default=42)
    device: str = Field(default="cuda")


class ModelConfig(BaseModel):
    name_or_path: str
    torch_dtype: str = Field(default="bfloat16")


class QuantizationConfig(BaseModel):
    formats: List[str] = Field(default_factory=lambda: ["fp16", "fp8", "nvfp4"])
    calibration_samples: int = Field(default=128)


class DatasetConfig(BaseModel):
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = Field(default="test")
    max_samples: Optional[int] = None


class EvaluationConfig(BaseModel):
    datasets: dict[str, DatasetConfig]
    batch_size: int = Field(default=1)


class OutputConfig(BaseModel):
    dir: str = Field(default="./outputs")
    save_plots: bool = Field(default=True)


class BenchmarkPipelineConfig(BaseModel):
    experiment: ExperimentConfig
    model: ModelConfig
    quantization: QuantizationConfig
    evaluation: EvaluationConfig
    output: OutputConfig