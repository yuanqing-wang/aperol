from setuptools import setup, find_packages

setup(
    name="aperol",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
    ],
)
