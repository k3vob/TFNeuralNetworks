import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="TFNeuralNetworks",
    version="0.3",
    author="Kevin O'Brien",
    author_email="kevin.vincent.obrien@gmail.com",
    description="A custom wrapper library for building highly encapsulated TensorFlow neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevOBrien/TFNeuralNetworks",
    packages=['TFNeuralNetworks'],
    install_requires=[
        'setuptools',
        'pandas',
        'tensorflow'
    ],
    licence='MIT'
)
