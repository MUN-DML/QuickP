# Efficient Device Placement for Distributed DNN Training

## Abstract
To pursue better predictive performance in deep neural networks (DNNs), the size of learning models tends to increase, 
resulting in high computation requirements and training latency. In this paper, we investigate the problem of device 
placement to accelerate large-scale distributed DNN training, but finding an effective scheme remains challenging due 
to its NP-hardness and the ever-increasing model size. To circumvent the high computational complexity of reinforcement 
learning schemes and the performance loss associated with heuristic schemes, we propose novel operator fusion and 
co-location schemes to reduce the search space while minimizing subsequent performance loss in training latency, 
enabling efficient device placement. We evaluate the performance of our design with real-world DNN benchmarks, and the 
results show that, compared to state-of-the-art approaches, our design achieves up to a 32\% reduction in DNN training 
latency and an order-of-magnitude improvement in placement search latency.

## Hardware dependency

## Installation

### Install Gurobi certificate. 
This solution requires the solver Gurobi to be installed and have an active license that students can apply for free

### Run steps
First, clone this project and the submodule of it from repo using SSH:

`git clone git@github.com:MUN-DML/QuickP.git`

Then, step into the ICC2025 folder:

`cd ICC2025`

Then, enter QuickP.py and adjust input parameters:
* `number_of_device` (integer): number of GPUs,
* `model` (string): DNN model name: `ALEXNET`, `VGG`, `FNET`, `BERT`
* `alpha` (integer): merging threshold for operator fusion,
