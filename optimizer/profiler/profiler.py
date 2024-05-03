import torch
import torch.nn as nn
import torchviz
from torch.profiler import profile, record_function, ProfilerActivity

from py_util import getStdModelForCifar10, getStdCifar10DataLoader
from resnet import ResNet18
from vgg import vgg11

# https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
# https://medium.com/computing-systems-and-hardware-for-emerging/profiling-a-training-task-with-pytorch-profiler-and-viewing-it-on-tensorboard-2cb7e0fef30e

model = vgg11().cuda()
trainloader = getStdCifar10DataLoader()

### optimizer, criterion ###
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)
### only one GPU per node, so we can directly use cuda() instead of .to()
criterion = nn.CrossEntropyLoss()
'''
Parameter skip_first tells profiler that it should ignore the first 10 steps (default value of skip_first is zero);
After the first skip_first steps, profiler starts executing profiler cycles;
Each cycle consists of three phases:
    idling (wait=5 steps), during this phase profiler is not active;
    warming up (warmup=1 steps), during this phase profiler starts tracing, but the results are discarded; this phase is used to discard the samples obtained by the profiler at the beginning of the trace since they are usually skewed by an extra overhead;
    active tracing (active=3 steps), during this phase profiler traces and records data;
    repeat parameter specifies an upper bound on the number of cycles. By default (zero value), profiler will execute cycles as long as the job runs.
'''
with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # in the following schedule, the profiler will record the performance form the 3 to the 8 mini-batch
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1),
        with_stack=True,
        with_flops=True,
        record_shapes=True,
        profile_memory=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
) as profiler:
    for step, data in enumerate(trainloader, 0):
        if step == 9:
            break
        print("step:{}".format(step))
        inputs, labels = data[0].cuda(), data[1].cuda()
        # forward
        with record_function("forward_pass"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        # backward
        with record_function("backward_pass"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # send a signal to the profiler that the next iteration has started
        profiler.step()

# Print the computation time of each operator
# print(profiler.key_averages().table(sort_by="cuda_time_total"))
# profiler.export_chrome_trace("result.json")
# torchviz.make_dot(outputs, params=dict(model.named_parameters())).render("computation_graph_forward", format="png")
for event in profiler.key_averages(group_by_stack_n=5):
    if 'cudaMemcpy' in event.name or 'cudaMemcpyAsync' in event.name:
        print(f"{event.name}: {event.cuda_time_total:.2f}us")