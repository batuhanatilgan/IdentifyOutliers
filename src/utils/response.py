from sdks.novavision.src.helper.package import PackageHelper
from components.IdentifyOutliers.src.models.PackageModel import (
    PackageModel,
    PackageConfigs,
    ConfigExecutor,
    IdentifyOutliersExecutor,
    IdentifyOutliersResponse,
    IdentifyOutliersOutputs,
    OutputIsOutlier,
    OutputPercentile,
    OutputWarmingUp,
)


def build_response(context):
    outputIsOutlier = OutputIsOutlier(value=context.is_outlier)
    outputPercentile = OutputPercentile(value=context.percentile)
    outputWarmingUp = OutputWarmingUp(value=context.warming_up)

    outputs = IdentifyOutliersOutputs(
        outputIsOutlier=outputIsOutlier,
        outputPercentile=outputPercentile,
        outputWarmingUp=outputWarmingUp,
    )

    identifyOutliersResponse = IdentifyOutliersResponse(outputs=outputs)
    identifyOutliersExecutor = IdentifyOutliersExecutor(value=identifyOutliersResponse)
    configExecutor = ConfigExecutor(value=identifyOutliersExecutor)
    packageConfigs = PackageConfigs(executor=configExecutor)

    package = PackageHelper(packageModel=PackageModel, packageConfigs=packageConfigs)
    return package.build_model(context)