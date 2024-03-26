import transformers
import tensor_parallel as tp

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/opt-13b")
model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-13b")  # use opt-125m for testing

# Simply wrap your PyTorch model with tp.tensor_parallel and use it normally.
# For best memory efficiency, call tp.tensor_parallel while the model is still on CPU.
'''
device_ids: List[device] - which devices to use; defaults to all available GPUs
output_device: device - model outputs will have this device
tensor_parallel_config: tp.Config - use custom parallelism strategy, see slicing_configs.py
distributed: bool - if True, use torch.distributed backend instead of threading (requires torchrun)
sharded: bool - if True, find all trainable parameters that weren't split by Tensor Parallelism and split them using ZeRO-3 algorithm.
'''
model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])  # <- each GPU has half the weights

inputs = tokenizer("A cat sat", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs, num_beams=5)
print(tokenizer.decode(outputs[0]))  # A cat sat on my lap for a few minutes ...

loss = model(input_ids=inputs, labels=inputs).loss  # training works as usual
loss.backward()  # check nvidia-smi for gpu memory usage :)
