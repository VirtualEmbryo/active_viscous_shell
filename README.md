# Active viscous shell

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3604085.svg)](https://doi.org/10.5281/zenodo.)

`active_viscous_shell` is distributed as a supplemental material to the paper:

> Borja da Rocha, H., Bleyer, J., & Turlier, H. (2021). A viscous active shell theory of the cell cortex. arXiv preprint arXiv:2110.12089.

The code is based on the [FEniCS software package](https://fenicsproject.org) for the finite-element implementation, [Gmsh](http://gmsh.info) for the meshing and the [Mmg platform](https://www.mmgtools.org/) for adaptive remeshing.

## Theory and implementation

See the [Theory and implementation](theory_implementation.md) document.

## Requirements

* **`FEniCS`** (>= 2019.1.0, Python 3), see [installation instructions here](https://fenicsproject.org/download/archive/).
* **`Gmsh`**, see [Download here](http://gmsh.info/download).
* the **`meshio`** package for mesh conversion (version 5.0 at least)
> ```
> pip install meshio==5.*
> ```

* the **`mmgs`** application of the MMG library, [Download here](http://www.mmgtools.org/mmg-remesher-downloads).
* 
## Usage

First, you should specify in the `paths.json` files the path corresponding to the `gmsh` and `mmgs_O3` application on your system such as:
```json
{
    "gmsh": "/path/to/gmsh",
    "mmg": "/path/to/mmgs_O3"
}
```


We use configuration files named `config.conf` for specifying the simulation parameters.
Each example is in the folder `./Examples`, including:
* `./Examples/HyperOsmoticShock`
* `./Examples/HypoOsmoticShock` 
* `./Examples/Cytokinesis`

To run one of the examples, first activate the FEniCS environment if you used a virtual environement based installation (e.g. `conda activate fenicsproject`). Then:
```
cd ./Examples/HyperOsmoticShock 
python3 ../../main.py
```
