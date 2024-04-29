import torch
import torch.nn as nn
import torchviz
from torch.autograd.profiler import record_function

from pyutil import getStdModelForCifar10, getStdCifar10DataLoader
from resnet import ResNet18

# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html

model = ResNet18().cuda()
trainloader = getStdCifar10DataLoader()

### optimizer, criterion ###
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
### only one GPU per node, so we can directly use cuda() instead of .to()
criterion = nn.CrossEntropyLoss().cuda()

with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as profiler:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].cuda(), data[1].cuda()
        # forward
        with record_function("model_inference"):
            outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # send a signal to the profiler that the next iteration has started
        profiler.step()

# Print the computation time of each operator
print(profiler.key_averages().table())

# torchviz.make_dot(outputs, params=dict(model.named_parameters())).render("computation_graph_forward", format="png")
