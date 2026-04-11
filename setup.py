from setuptools import setup, find_packages

setup(
    name="aperol",
    version="0.1.0",
    description="Equivariant neural networks for molecular force fields (MD17)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
    ],
    extras_require={
        "train": ["wandb"],
    },
)
