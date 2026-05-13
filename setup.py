import setuptools

setuptools.setup(
    name="identify-outliers",
    version="0.0.1",
    author="DigiNova",
    author_email="info@diginova.com.tr",
    description="Detects outlier embeddings using von Mises-Fisher distribution over a sliding window of historical SIFT descriptor vectors.",
    url="https://github.com/novavision-ai/identify-outliers",
    license="MIT",
    install_requires=["sdk", "numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=[
        "novavision.identify_outliers",
        "novavision.identify_outliers.classes",
        "novavision.identify_outliers.configs",
        "novavision.identify_outliers.dataloaders",
        "novavision.identify_outliers.executors",
        "novavision.identify_outliers.models",
        "novavision.identify_outliers.utils",
        "novavision.identify_outliers.weights",
    ],
    package_dir={"novavision.identify_outliers": "src"},
    python_requires=">=3.6",
)