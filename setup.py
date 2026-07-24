from setuptools import setup, find_packages

setup(
    name="wiki",
    version="1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": ["wiki=pipeline.cli:cli"],
    },
)
