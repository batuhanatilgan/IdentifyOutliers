from sdks.novavision.src.helper.package import PackageHelper
from components.IdentifyOutliers.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    IdentifyOutliersExecutor,
    IdentifyOutliersResponse,
    IdentifyOutliersOutputs,
    OutputDetections
)

def build_response(context):
    outputDetections = OutputDetections(value=context.detections)

    outputs = IdentifyOutliersOutputs(
        outputDetections=outputDetections
    )

    identifyOutliersResponse = IdentifyOutliersResponse(outputs=outputs)
    identifyOutliersExecutor = IdentifyOutliersExecutor(value=identifyOutliersResponse)
    configExecutor = ConfigExecutor(value=identifyOutliersExecutor)
    packageConfigs = PackageConfigs(executor=configExecutor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)