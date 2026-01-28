import os
import setuptools


def read_requirements():
    try:
        with open(os.path.dirname(__file__) + "/requirements.txt", "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        print("requirements.txt not found. Proceeding without it.")
        return []


setuptools.setup(
    name="noqta",
    version="0.0.1",
    author="Khaled Ibrahim",
    author_email="khaled.ibrahim@crowdlinker.com",
    description="Python package for document layout chunking using traditional computer vision.",
    url="https://github.com/Khaledhamza77/Noqta",
    packages=setuptools.find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
)
