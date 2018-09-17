import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="TFNeuralNetworks",
    version="0.0.1",
    author="Kevin O'Brien",
    author_email="kevin.vincent.obrien@gmail.com",
    description="A custom wrapper library for building highly encapsulated TensorFlow neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevOBrien/TFNeuralNetworks",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
