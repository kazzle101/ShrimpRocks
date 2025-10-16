#!/usr/bin/env python3
"""
Packaging metadata for the ShrimpRocks project.
"""

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).resolve().parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name="shrimp-rocks",
    version="0.1.0",
    description="Automated analysis toolkit for a Chesil Beach pebble survey.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="kazzle101",
    url="https://github.com/kazzle101/ShrimpRocks",
    packages=find_packages(exclude=("test", "tests", "images", "scripts")),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.6",
        "natsort>=8.0",
        "numpy>=1.23",
        "opencv-python>=4.7",
        "pillow>=10.0",
        "segment-anything>=1.0",
        "torch>=2.0",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "shrimp-rocks=shrimpRocks.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
