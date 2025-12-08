<!--intro-start-->
# heiplanet-data: A data processing pipeline for climate and population data

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ssciwr/heiplanet-data/main.svg)](https://results.pre-commit.ci/latest/github/ssciwr/heiplanet-data/main)
![License: MIT](https://img.shields.io/github/license/ssciwr/heiplanet-data)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/heiplanet-data/ci.yml?branch=main)
[![codecov](https://codecov.io/gh/ssciwr/heiplanet-data/graph/badge.svg?token=26SVxiuDO7)](https://codecov.io/gh/ssciwr/heiplanet-data)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_onehealth-data-backend&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=ssciwr_onehealth-data-backend)
![Language](https://img.shields.io/github/languages/top/ssciwr/heiplanet-data)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ssciwr/heiplanet-data/blob/main/heiplanet-data/docs/source/notebooks/tutorial_A_download_data.ipynb)

A data processing pipeline for Copernicus and Eurostat data. Please visit the [documentation](https://ssciwr.github.io/heiplanet-data/) for further information.

## Installation

The package currently supports Python 3.11 to 3.13

* Create a `conda` environment
    ```bash
    conda create -n heiplanet-data python=3.13 -y
    conda init bash
    source ~/.bashrc
    conda activate heiplanet-data
    ```
* Install the package from the repository
    * From the root directory of the repository (with your `conda` environment activated), run:
    `pip install .`

* Install `esmf` and `esmpy` from `conda-forge` for `xESMF` package to perform downsampling
    `conda install -c conda-forge esmf esmpy`

* Install `cdo` python package from `conda-forge` in case of using `cdo` for downsampling
    `conda install -c conda-forge python-cdo`

## Further information

* [About](docs/about.md): Information about the project, research group, and development group
* [Data](docs/data.md): Used data sources and conventions
* [Datalake](docs/datalake.md): Data lake architecture and files
* [Tutorials](docs/source/notebooks/tutorial_A_download_data.ipynb): Tutorials how to use heiplanet-data ([Tutorial A: Download data (open in colab)][tutorial A], [Tutorial B: Process data (open in colab)][tutorial B], [Tutorial C: Aggregate data (open in colab)][tutorial C])
* [Issues](docs/issues.md): Some notes for running the package and further issues
* [Contact](docs/contact.md): Contact information of relevant groups
* [License](docs/license.md): License information

[tutorial A]: https://colab.research.google.com/github/ssciwr/heiplanet-data/blob/main/heiplanet-data/docs/source/notebooks/tutorial_A_download_data.ipynb 
[tutorial B]: https://colab.research.google.com/github/ssciwr/heiplanet-data/blob/main/heiplanet-data/docs/source/notebooks/tutorial_B_preprocess_data.ipynb
[tutorial C]: https://colab.research.google.com/github/ssciwr/heiplanet-data/blob/main/heiplanet-data/docs/source/notebooks/tutorial_C_postprocess_data.ipynb


<!--intro-end-->
