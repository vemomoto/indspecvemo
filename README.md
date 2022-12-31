# IndSpecVeMo - an Individual-Specific Vector Model

This repository contains the code for the individual-specific vector model by Fischer, Ramazi, Simmons, Poesch and Lewis (see https://arxiv.org/abs/2105.14284). 

The methods partially build on the libdatrie package (https://github.com/pytries/datrie) and hence the repository includes a snapshot of the project's code.

**Caution:** this is an as-is version of the code from the research project. That is, the code is not yet fully cleaned up and documented. 

## Installation

To build the indspecvemo extension and use the included model 
and tools, this setup file must be run. We assume that the 
python package manager pip (see https://pypi.org/project/pip/) 
is installed and ready to use.

We also assume that a Python-compatible C++ compiler is installed
and in the PATH. This is typically the case on Unix systems. 
On Windows, you may need to install the VisualStudio Build Tools
and run the setup from the "Developer Command Prompt for VS".
The compiler is necessary, because parts of the model rely on 
functions written in C++.

The package can be installed by navigating in the console to the
folder containing this file (setup.py). Run 

```
pip install .
```

to install the package and its dependencies.

## Usage

An example of the model usage can be found in the file example.py
Please go through the 


