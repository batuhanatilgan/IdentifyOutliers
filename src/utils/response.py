from sdks.novavision.src.helper.package import PackageHelper
from components.IdentifyOutliers.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    IdentifyOutliersExecutor,
    IdentifyOutliersResponse,
    IdentifyOutliersOutputs,
    OutputData
)

def build_response(context):
    outputData = OutputData(value=context.output_data)

    outputs = IdentifyOutliersOutputs(
        outputData=outputData
    )

    identifyOutliersResponse = IdentifyOutliersResponse(outputs=outputs)
    identifyOutliersExecutor = IdentifyOutliersExecutor(value=identifyOutliersResponse)
    configExecutor = ConfigExecutor(value=identifyOutliersExecutor)
    packageConfigs = PackageConfigs(executor=configExecutor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)