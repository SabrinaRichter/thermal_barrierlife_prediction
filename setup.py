from setuptools import setup, find_packages

author = 'Sabrina Richter'
author_email = 'sabrina.richter@helmholtz-muenchen.de'
description = ""
__version__ = "0.0.0"

with open("README.md", "r") as fh:
     long_description = fh.read()

setup(
    name='thermal_barrierlife_prediction',
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[],
    version=__version__,
)
