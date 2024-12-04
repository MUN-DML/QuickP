# Efficient Device Placement for Distributed DNN Training

## Installation

### Install Gurobi certificate. 
This solution requires the solver Gurobi to be installed and have an active license that students can apply for free

### Run steps
First, clone this project and the submodule of it from repo using SSH:

`git clone git@github.com:MUN-DML/QuickP.git`

Then, step into the ICC2025 folder:

`cd ICC2025`

Configure the environment variable for Gurobi license

`os.environ['GRB_LICENSE_FILE'] = 'your_path/gurobi.lic'`

Then, enter QuickP.py and adjust input parameters:
* `number_of_device` (integer): number of GPUs,
* `model` (string): DNN model name: `ALEXNET`, `VGG`, `FNET`, `BERT`
* `alpha` (integer): merging threshold for operator fusion,

Finally, make sure your current working directory is ICC2025 and run QuickP.py

`python QuickP.py`
