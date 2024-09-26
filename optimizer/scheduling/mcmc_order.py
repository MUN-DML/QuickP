# MCMC searching logic is implemented in MCMC.py. Here is to apply the sequential order constraint
from gurobipy import Model


def mcmc_schedule(model: Model, start, finish, comm_start, comm_end, device_subgraph_mapping: dict,
                  operator_device_mapping: dict, mcmc_order_dict):
    pass
