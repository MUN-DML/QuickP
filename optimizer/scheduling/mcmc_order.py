# MCMC searching logic is implemented in MCMC.py. Here is to apply the sequential order constraint
from gurobipy import Model


def mcmc_schedule(model: Model, start, finish, comm_start, comm_end, mcmc_order_dict: dict):
    for sequence in mcmc_order_dict.items():
        for a, b in zip(sequence, sequence[1:]):
            model.addConstr(finish[a] <= start[b])
