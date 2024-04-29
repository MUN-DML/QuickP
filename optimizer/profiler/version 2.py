import torch
import torch.nn as nn
import torchviz

from pyutil import getStdModelForCifar10, getStdCifar10DataLoader

model = getStdModelForCifar10().cuda()
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
        profile_memory=True
) as profiler:
    for step, data in enumerate(trainloader, 0):
        print("step:{}".format(step))
        inputs, labels = data[0].cuda(), data[1].cuda()
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # send a signal to the profiler that the next iteration has started
        profiler.step()

# Print the computation time of each operator
print(profiler.key_averages().table(sort_by="self_cuda_memory_usage"))
profiler.export_stacks(path="./profiler.json", metric="self_cuda_time_total")
# torchviz.make_dot(outputs, params=dict(model.named_parameters())).render("computation_graph_forward", format="png")
