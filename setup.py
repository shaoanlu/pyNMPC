# setup.py
from setuptools import setup, find_packages

setup(
    name="pyNMPC",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "cvxpy[OSQP, PIQP]>=1.4.2",
        "jax>=0.5.0",
        "numpy>=1.26.4",
    ],
    author="shaoanlu",
    author_email="abc@email.com",
    description="Nonlinear Model Predictive Control based on CVXPY and JAX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/shaoanlu/pyNMPC",
    license="Apache License 2.0",
    python_requires=">=3.10",
)
