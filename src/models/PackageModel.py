from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Detection, Inputs, Configs, Outputs,
    Response, Request, Output, Input, Config
)

class InputDetections(Input):
    """The incoming SIFT detections containing descriptor vectors to be analyzed for anomalies."""
    name: Literal["inputDetections"] = "inputDetections"
    value: Union[List[Detection], Detection]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        v = values.get("value")
        if isinstance(v, list):
            return "list"
        return "object"

    class Config:
        title = "Detections"

class OutputDetections(Output):
    """The output detection data enriched with the 'Identify' outlier analysis results."""
    name: Literal["outputDetections"] = "outputDetections"
    value: List[Detection]
    type: Literal["list"] = "list"

    class Config:
        title = "Detections"

class ConfigThresholdPercentile(Config):
    """
    Percentile threshold for outlier detection, range 0.0 to 1.0.
    Lower values detect only extreme outliers. Higher values are more sensitive.
    """
    name: Literal["ConfigThresholdPercentile"] = "ConfigThresholdPercentile"
    value: float = Field(ge=0.0, le=1.0, default=0.05)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Threshold Percentile"
        json_schema_extra = {"shortDescription": "Outlier sensitivity threshold (0.0 to 1.0)."}

class ConfigWarmup(Config):
    """
    Number of initial samples required before outlier detection begins.
    During this period all frames return is_outlier=False to allow baseline establishment.
    """
    name: Literal["ConfigWarmup"] = "ConfigWarmup"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Warmup Samples"
        json_schema_extra = {"shortDescription": "Minimum samples collected before detection starts."}

class ConfigWindowSize(Config):
    """
    Maximum number of historical embeddings stored in the sliding window.
    When exceeded, the oldest embedding is removed.
    """
    name: Literal["ConfigWindowSize"] = "ConfigWindowSize"
    value: int = Field(ge=2, default=32)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Window Size"
        json_schema_extra = {"shortDescription": "Number of recent embeddings kept for comparison."}

class IdentifyOutliersInputs(Inputs):
    """Aggregates the inputs required for the Identify Outliers package."""
    inputDetections: InputDetections

class IdentifyOutliersConfigs(Configs):
    """Aggregates the configuration parameters for the Identify Outliers package."""
    configThresholdPercentile: ConfigThresholdPercentile
    configWarmup: ConfigWarmup
    configWindowSize: ConfigWindowSize

class IdentifyOutliersOutputs(Outputs):
    """Aggregates the outputs produced by the Identify Outliers package."""
    outputDetections: OutputDetections

class IdentifyOutliersRequest(Request):
    """The incoming request payload containing inputs and configurations."""
    inputs: Optional[IdentifyOutliersInputs]
    configs: IdentifyOutliersConfigs

    class Config:
        json_schema_extra = {"target": "configs"}

class IdentifyOutliersResponse(Response):
    """The outgoing response payload containing the enriched detections."""
    outputs: IdentifyOutliersOutputs

class IdentifyOutliersExecutor(Config):
    """The main executor that detects outlier embeddings using von Mises-Fisher directional statistics."""
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"
    value: Union[IdentifyOutliersRequest, IdentifyOutliersResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Identify Outliers"
        json_schema_extra = {"target": {"value": 0}}

class ConfigExecutor(Config):
    """Top-level selection for the model task and execution logic."""
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[IdentifyOutliersExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}

class PackageConfigs(Configs):
    """Wraps the top-level executor configuration for the package."""
    executor: ConfigExecutor

class PackageModel(Package):
    """The root model defining the Identify Outliers component package structure."""
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"