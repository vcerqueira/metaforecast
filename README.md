# metaforecast

[![PyPi Version](https://img.shields.io/pypi/v/metaforecast)](https://pypi.org/project/metaforecast/)
[![Documentation Status](https://readthedocs.org/projects/metaforecast/badge/?version=latest)](https://metaforecast.readthedocs.io/en/latest/?badge=latest)
[![GitHub](https://img.shields.io/github/stars/vcerqueira/metaforecast?style=social)](https://github.com/vcerqueira/metaforecast)
![Pylint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/vcerqueira/7ad63bc9902a43eb21993a755006f5de/raw/pylint-badge.json)
[![Downloads](https://static.pepy.tech/badge/metaforecast)](https://pepy.tech/project/metaforecast)


metaforecast is a Python package for time series forecasting using meta-learning and data-centric techniques.

This package implements various techniques to improve forecasting accuracy 
based on dynamic model combination, data augmentation, and adaptive learning, building upon Nixtla’s awesome ecosystem of state-of-the-art forecasting methods.

## Features

metaforecast currently consists of three main modules:

1. **Dynamic Ensembles**: Combining multiple models with adaptive ensemble techniques.
2. **Synthetic Time Series Generation**: Creating realistic synthetic time series data for robust model training and testing. 
Includes a special callback for online data augmentation.
3. **Long-Horizon Meta-Learning**: Instance-based meta-learning for multi-step forecasting.


## Installation

You can install metaforecast using pip:

```bash
pip install metaforecast
```

### [Optional] Installtion from source

To install metaforecast from source, clone the repository and run the following command:

```bash
git clone https://github.com/vcerqueira/metaforecast
pip install -e metaforecast
cd metaforecast
pre-commit install
```

## Documentation

Check the [documentation](https://metaforecast.readthedocs.io/en/latest/index.html) for 
the API reference and module descriptions. 
You can get started with a few [tutorials](https://metaforecast.readthedocs.io/en/latest/notebooks.html).

----

### **⚠️ WARNING**

> metaforecast is in the early stages of development. 
> The codebase may undergo significant changes. 
> If you encounter any issues, please report
> them in [GitHub Issues](https://github.com/vcerqueira/metaforecast/issues)
