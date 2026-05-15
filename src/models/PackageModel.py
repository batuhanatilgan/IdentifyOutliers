from pydantic import Field, validator
from typing import List, Optional, Union, Literal, Dict, Any
from sdks.novavision.src.base.model import (
    Package, Inputs, Configs, Outputs,
    Response, Request, Output, Input, Config
)

class InputData(Input):
    """The incoming data containing pre-calculated embeddings (e.g., from CLIP)."""
    name: Literal["inputData"] = "inputData"
    value: Union[List[Dict], Dict, List, Any]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        v = values.get("value")
        if isinstance(v, list):
            return "list"
        return "object"

    class Config:
        title = "Input Data"

class OutputData(Output):
    """The output data enriched with the 'Identify' outlier analysis results."""
    name: Literal["outputData"] = "outputData"
    value: Union[List[Dict], Dict, List, Any]
    type: Literal["list"] = "list"

    class Config:
        title = "Output Data"

class ConfigThresholdPercentile(Config):
    name: Literal["ConfigThresholdPercentile"] = "ConfigThresholdPercentile"
    value: float = Field(ge=0.0, le=1.0, default=0.05)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[0.0, 1.0]"] = "[0.0, 1.0]"

    class Config:
        title = "Threshold Percentile"
        json_schema_extra = {"shortDescription": "Outlier sensitivity threshold (0.0 to 1.0)."}

class ConfigWarmup(Config):
    name: Literal["ConfigWarmup"] = "ConfigWarmup"
    value: int = Field(ge=2, default=10)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Warmup Samples"
        json_schema_extra = {"shortDescription": "Minimum samples collected before detection starts."}

class ConfigWindowSize(Config):
    name: Literal["ConfigWindowSize"] = "ConfigWindowSize"
    value: int = Field(ge=2, default=32)
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"
    placeHolder: Literal["[2, ...]"] = "[2, ...]"

    class Config:
        title = "Window Size"
        json_schema_extra = {"shortDescription": "Number of recent embeddings kept for comparison."}

class IdentifyOutliersInputs(Inputs):
    inputData: InputData

class IdentifyOutliersConfigs(Configs):
    configThresholdPercentile: ConfigThresholdPercentile
    configWarmup: ConfigWarmup
    configWindowSize: ConfigWindowSize

class IdentifyOutliersOutputs(Outputs):
    outputData: OutputData

class IdentifyOutliersRequest(Request):
    inputs: Optional[IdentifyOutliersInputs]
    configs: IdentifyOutliersConfigs

    class Config:
        json_schema_extra = {"target": "configs"}

class IdentifyOutliersResponse(Response):
    outputs: IdentifyOutliersOutputs

class IdentifyOutliersExecutor(Config):
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"
    value: Union[IdentifyOutliersRequest, IdentifyOutliersResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Identify Outliers"
        json_schema_extra = {"target": {"value": 0}}

class ConfigExecutor(Config):
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[IdentifyOutliersExecutor]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"target": "value"}

class PackageConfigs(Configs):
    executor: ConfigExecutor

class PackageModel(Package):
    configs: PackageConfigs
    type: Literal["component"] = "component"
    name: Literal["IdentifyOutliers"] = "IdentifyOutliers"