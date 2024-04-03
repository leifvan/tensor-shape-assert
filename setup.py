# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="tensor-shape-assert",
    version="0.0.1",
    description="A simple runtime assert library for tensor-based frameworks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leifvan/tensor-shape-assert",
    author="Leif Van Holland",
    author_email="holland@cs.uni-bonn.de",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Unlicense",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="sample, setuptools, development",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7, <4"
)