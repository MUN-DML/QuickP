import argparse
import os

import torchvision
from pippy import annotate_split_points, SplitPoint
from tensorflow.python.framework.dtypes import DType
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms

# This guide can only be run with the torch backend. must write when using both keras and pytorch
# sudo apt install python3-packaging
os.environ["KERAS_BACKEND"] = "torch"

import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader


# our module must be nn.Sequential as GPipe will automatically split the module into partitions with consecutive layers
# The previous layer's out_channels should match the next layer's in_channels
# https://stackoverflow.com/questions/68606661/what-is-difference-between-nn-module-and-nn-sequential;
# using nn-module or nn-sequential
def getStdModelForCifar10():
    return nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        # the out_features number of the last layer should match the class number
        nn.Linear(84, 10))


# Data loading code for CiFar10
def getStdCifar10DataLoader(batch_size=256, num_workers=0, train=True):
    """
    If Use keras dataset instead of torchvision
    https://keras.io/guides/writing_a_custom_training_loop_in_torch/
    """
    # Data loading code for CiFar10
    transform_train = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=train, transform=transform_train, download=True)
    # sampler=train_sampler; if sampler is defined, set the shuffle to false
    return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                       pin_memory=True, num_workers=num_workers)


def saveModelState(model, modelName):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = "./model/{}".format(modelName)
    torch.save(model.state_dict(), filepath)


def testPYModel(model, test_loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for (inputs, targets) in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def getArgs():
    parser = argparse.ArgumentParser(description='cifar10 classification models, single node model parallelism test')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=200, help='')
    parser.add_argument('--epochs', type=int, default=1, help='')
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--num_workers', type=int, default=-1, help='')
    parser.add_argument('--world_size', default=-1, type=int, help='')
    # 'tcp://192.168.0.66:3456'
    parser.add_argument('--init_method', default='env://', type=str, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--master_addr', type=str, default=os.getenv('MASTER_ADDR', '192.168.0.66'))
    parser.add_argument('--master_port', type=str, default=os.getenv('MASTER_PORT', '3456'))
    parser.add_argument('--chunks', type=int, default=4)
    args = parser.parse_args()
    nodeID = int(os.environ.get("SLURM_NODEID"))
    # DDP setting
    # update world size, rank, and if distributed in the args
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:  # for slurm scheduler
        # for homo platform where each node has the same number of GPU
        args.world_size = int(os.environ["SLURM_NTASKS_PER_NODE"]) * int(os.environ["SLURM_JOB_NUM_NODES"])
    args.distributed = args.world_size > 1

    if 'SLURM_LOCALID' in os.environ:
        args.local_rank = int(os.environ.get("SLURM_LOCALID"))

    if args.distributed:
        if 'SLURM_PROCID' in os.environ:  # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            # args.gpu = args.rank % torch.cuda.device_count()
        else:
            ngpus_per_node = torch.cuda.device_count()
            args.rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + args.local_rank

    args.gpu = args.local_rank

    if 'SLURM_CPUS_PER_TASK' in os.environ:
        args.num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])

    nodeName = os.environ.get("SLURMD_NODENAME")

    print("nodeID: ", nodeID, "nodeName: ", nodeName, " distributed mode: ", args.distributed, " from rank: ",
          args.rank,
          " world_size: ", args.world_size, " num_workers: ", args.num_workers, " local_rank(always 0): ",
          args.local_rank, " gpu(always 0): ", args.gpu)
    return args


# set the env variables of rank and world_size
def setup(rank, world_size):
    os.environ["RANK"] = rank
    os.environ["WORLD_SIZE"] = world_size


def retrieve_existing_model(obj, modelName):
    # Prepare a directory to store all the checkpoints.
    checkpoint_dir = "./model"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = "./model/{}".format(modelName)
    obj.load_state_dict(torch.load(filepath))
    obj.eval()
    return obj


def printPipelineSplitInfo(rank, pipe):
    if rank == 0:
        print(" pipe ".center(80, "*"))
        print(pipe)
        for i, sm in enumerate(pipe.split_gm.children()):
            print(" stage {} ".format(i).center(80, "*"))
            print(sm)


def train(epoch, train_loader, model, criterion, optimizer, device, profile_one_itr=False):
    # only one gpu is visible here, so you can send cpu data to gpu by
    # input_data = input_data.cuda() as normal
    train_loss = 0
    correct = 0
    total = 0
    model.train(True)
    profiler_schedule = torch.profiler.schedule(wait=2, warmup=2, active=1, repeat=1) if profile_one_itr else None
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=profiler_schedule
    ) as prof:
        for i in range(epoch):
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx == 5 and profile_one_itr:
                    break
                inputs = inputs.to(device)
                # shape: batch_size * 1
                targets = targets.to(device)
                # optimizer.zero_grad() everywhere in the loop but not between the loss.backward() and optimizer.step()
                optimizer.zero_grad()
                # shape batch_size * 10
                outputs = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()

                optimizer.step()

                train_loss += loss.item()
                _, predicted_idx = outputs.max(1)
                total += targets.size(0)
                correct += predicted_idx.eq(targets).sum().item()
                acc = correct / total
                if batch_idx % 20 == 0:  # print every 20 mini-batches
                    print(f'[{i}, {batch_idx}] loss: {train_loss / 2000:.3f} accu: {acc * 100:.2f}')
                    train_loss = 0.0
                # send a signal to the profiler that the next iteration has started
                prof.step()
    # prof.export_chrome_trace("single_device_prof.json")
    print(prof.key_averages().table(sort_by="cuda_time_total"))


def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(logits, torch.distributed.rpc.api.RRef):
                logits = logits.local_value()
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def get_number_of_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_split_points(model, world_size):
    for i in range(1, world_size):
        # the name should correspond to the layer name in the model
        annotate_split_points(
            model, {f"layer{i}": SplitPoint.BEGINNING})


def print_communication_cost(table_str):
    # Split the table into lines
    lines = table_str.split('\n')

    # Initialize a list to hold the titles
    filtered_lines = lines[0:3]

    # Search for rows that contain the keyword 'AllReduce'
    for line in lines[3:]:
        if 'all_reduce' in line:
            filtered_lines.append(line)

    filtered_lines = filtered_lines + lines[-3:]

    for line in filtered_lines:
        print(line)


def tensor_shape_to_bits(tensor_shape, dtype: DType):
    """
    Convert a TensorShape to the total number of bits.

    :param tensor_shape: A TensorShape object or a list/tuple of dimensions.
    :param dtype: The data type of the tensor elements (e.g., tf.float32, tf.float64).
    :return: Total size in bits.
    """
    # Get the total number of elements
    total_elements = tf.reduce_prod(tensor_shape).numpy()

    # Determine the number of bits per element based on dtype
    if dtype == tf.float32 or dtype == tf.int32:
        bits_per_element = 32
    elif dtype == tf.float64 or dtype == tf.int64:
        bits_per_element = 64
    elif dtype == tf.float16:
        bits_per_element = 16
    elif dtype == tf.int16:
        bits_per_element = 16
    elif dtype == tf.int8 or dtype == tf.uint8:
        bits_per_element = 8
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    # Calculate the total size in bits
    total_bits = total_elements * bits_per_element
    return total_bits


def convert_data_size(value, from_unit, to_unit):
    """
    Convert between different data size units.

    :param value: The numerical value to convert.
    :param from_unit: The unit of the input value (e.g., 'GB', 'Mb', 'bytes', etc.).
    :param to_unit: The unit to convert to (e.g., 'GB', 'Mb', 'bytes', etc.).
    :return: The converted value.
    """

    # Define conversion factors relative to bytes
    conversion_factors = {
        'bit': 1 / 8,
        'kb': 1000,
        'kB': 1000 * 8,
        'Kb': 1024,
        'KB': 1024 * 8,
        'Mb': 1024 ** 2,
        'MB': 1024 ** 2 * 8,
        'Gb': 1024 ** 3,
        'GB': 1024 ** 3 * 8,
        'Tb': 1024 ** 4,
        'TB': 1024 ** 4 * 8,
        'Pb': 1024 ** 5,
        'PB': 1024 ** 5 * 8,
        'bytes': 1,
        'B': 1,
    }


def convert_time(value, from_unit, to_unit):
    """
    Convert between different time units.

    :param value: The numerical value to convert.
    :param from_unit: The unit of the input value (e.g., 's', 'ms', 'min', etc.).
    :param to_unit: The unit to convert to (e.g., 's', 'ms', 'min', etc.).
    :return: The converted value.
    """

    # Define conversion factors relative to seconds
    conversion_factors = {
        'ns': 1e-9,  # Nanoseconds to seconds
        'µs': 1e-6,  # Microseconds to seconds
        'us': 1e-6,  # Microseconds to seconds (alternative symbol)
        'ms': 1e-3,  # Milliseconds to seconds
        's': 1,  # Seconds to seconds
        'min': 60,  # Minutes to seconds
        'h': 3600,  # Hours to seconds
        'd': 86400,  # Days to seconds
        'w': 604800  # Weeks to seconds
    }

    # Convert the input value to seconds
    seconds_value = value * conversion_factors[from_unit]

    # Convert the seconds value to the target unit
    converted_value = seconds_value / conversion_factors[to_unit]

    return converted_value
