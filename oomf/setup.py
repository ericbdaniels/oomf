import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="oomf",
    author="Eric Daniels",
    description="Obvious Open Mining Format",
    packages=["oomf"],
    version="0.0.1",
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/ericbdaniels/Oomf",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.19.2" "omf==1.0.1",
        "omfvista==0.2.2",
        "rich>=9.0.0",
    ],
    python_requires=">=3.6",
)
