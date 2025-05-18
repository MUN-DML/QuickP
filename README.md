# Efficient Device Placement for Distributed DNN Training (ICC 2025)


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14511161.svg)](https://doi.org/10.5281/zenodo.14511161)


## Installation

### Install Gurobi certificate. 
This solution requires the solver Gurobi to be installed and have an active license that students can apply for free

### Dependency
Details are listed inside this project

### Run steps
First, clone this project and the submodule of it from repo using SSH:

`git clone git@github.com:MUN-DML/QuickP.git`

Then, step into the ICC2025 folder:

`cd ICC2025`

Configure the environment variable for Gurobi license

`os.environ['GRB_LICENSE_FILE'] = 'your_path/gurobi.lic'`

Then, enter QuickP.py and adjust input parameters:
* `number_of_device` (integer): number of GPUs,
* `model` (string): DNN model name: `ALEXNET`, `VGG`, `FNET`, `BERT`,
* `beta` (integer): WCC expanding threshold,
* `alpha` (integer): merging threshold for operator fusion. Find the line "alpha = get_proper_alpha(comp_graph, deviceTopo, if_visualize=False)" and replace the function with your self-defined value,

Finally, make sure your current working directory is ICC2025 and run QuickP.py

`python QuickP.py`
