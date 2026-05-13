from pydantic import Field, validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Detection, Inputs, Configs, Outputs,
    Response, Request, Output, Input, Config
)


class InputDetections(Input):
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


class OutputIsOutlier(Output):
    name: Literal["outputIsOutlier"] = "outputIsOutlier"
    value: bool
    type: Literal["bool"] = "bool"

    class Config:
        title = "Is Outlier"


class OutputPercentile(Output):
    name: Literal["outputPercentile"] = "outputPercentile"
    value: float
    type: Literal["number"] = "number"

    class Config:
        title = "Percentile"


class OutputWarmingUp(Output):
    name: Literal["outputWarmingUp"] = "outputWarmingUp"
    value: bool
    type: Literal["bool"] = "bool"

    class Config:
        title = "Warming Up"


class ConfigThresholdPercentile(Config):
    """
    Percentile threshold for outlier detection, range 0.0 to 1.0.
    Embeddings below this percentile or above (1 - threshold) are flagged as outliers.
    Lower values (e.g. 0.01) detect only extreme outliers. Higher values (e.g. 0.1) are more sensitive.
    """
    name: Literal["ConfigThresholdPercentile"] = "ConfigThresholdPercentile"
    value: float = Field(ge=0.0, le=1.0, default=0.05)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Threshold Percentile"
        json_schema_extra = {
            "shortDescription": "Outlier sensitivity threshold (0.0 to 1.0)."
        }


class ConfigWarmup(Config):
    """
    Number of initial samples required before outlier detection begins.
    During this period all frames return is_outlier=False to allow baseline establishment.
    Must be at least 2. Typical range: 3 to 100.
    """
    name: Literal["ConfigWarmup"] = "ConfigWarmup"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Warmup Samples"
        json_schema_extra = {
            "shortDescription": "Minimum samples collected before detection starts."
        }


class ConfigWindowSize(Config):
    """
    Maximum number of historical embeddings stored in the sliding window.
    When exceeded, the oldest embedding is removed (FIFO).
    Larger windows yield more stable statistics but adapt slower to distribution changes.
    """
    name: Literal["ConfigWindowSize"] = "ConfigWindowSize"
    value: int = Field(ge=2, default=32)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Window Size"
        json_schema_extra = {
            "shortDescription": "Number of recent embeddings kept for comparison."
        }


class IdentifyOutliersInputs(Inputs):
    inputDetections: InputDetections


class IdentifyOutliersConfigs(Configs):
    configThresholdPercentile: ConfigThresholdPercentile
    configWarmup: ConfigWarmup
    configWindowSize: ConfigWindowSize


class IdentifyOutliersOutputs(Outputs):
    outputIsOutlier: OutputIsOutlier
    outputPercentile: OutputPercentile
    outputWarmingUp: OutputWarmingUp


class IdentifyOutliersRequest(Request):
    inputs: Optional[IdentifyOutliersInputs]
    configs: IdentifyOutliersConfigs

    class Config:
        json_schema_extra = {
            "target": "configs"
        }


class IdentifyOutliersResponse(Response):
    outputs: IdentifyOutliersOutputs


class IdentifyOutliersExecutor(Config):
    """
    Detects outlier embeddings using von Mises-Fisher directional statistics
    over a sliding window of historical SIFT descriptor vectors.
    """
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"
    value: Union[IdentifyOutliersRequest, IdentifyOutliersResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Identify Outliers"
        json_schema_extra = {
            "target": {
                "value": 0
            }
        }


class ConfigExecutor(Config):
    """
    Task selector that defines which execution component will run within this package.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[IdentifyOutliersExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {
            "target": "value"
        }


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"