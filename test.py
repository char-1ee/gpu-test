import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import argparse
import os
import time
import numpy as np


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.time.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def cleanup():
    dist.destroy_process_group()

def test_bandwidth(rank, world_size, dtype, op):
    # setup(rank, world_size)

    # Create a tensor
    tensor_size = 1024 * 1024 * 256  # 256 MB
    tensor = torch.rand(tensor_size, dtype=dtype, device="cuda")

    # Warm-up
    for _ in range(10):
        if op == "all-reduce":
            dist.all_reduce(tensor)
        elif op == "all-gather":
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
        elif op == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op == "all-to-all":
            tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_to_all(tensors, [tensor] * world_size)

    # Benchmark
    start_time = time.time()
    for _ in range(50):
        if op == "all-reduce":
            dist.all_reduce(tensor)
        elif op == "all-gather":
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
        elif op == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op == "all-to-all":
            tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_to_all(tensors, [tensor] * world_size)

    duration = time.time() - start_time
    bandwidth = (tensor.nelement() * tensor.element_size() * 50) / duration / 1e9

    if rank == 0:
        print(f"Operation: {op}, Data type: {dtype}, Bandwidth (GB/s): {bandwidth}")

    cleanup()

# def main():
#     parser = argparse.ArgumentParser(description="PyTorch Distributed Bandwidth Test")
#     parser.add_argument("--")
#     parser.add_argument("--world-size", type=int, default=4)
#     parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
#     parser.add_argument("--op", default="all-reduce", choices=["all-reduce", "all-gather", "broadcast", "all-to-all"])

#     args = parser.parse_args()
#     dtype = torch.float32 if args.dtype == "float32" else torch.float16
#     op = args.op

#     mp.spawn(
#         test_bandwidth,
#         args=(args.world_size, dtype, op),
#         nprocs=args.world_size,
#         join=True,
#     )


def main():
    parser = argparse.ArguementParser(description="Distributed GPU Bandwidth Test")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float16"]
    )
    parser.add_argument(
        "--op",
        type=str
        default="all-reduce",
        choices=["all-reduce", "all-gather", "broadcast", "all-to-all"]
    )
    args = parser.parse_args()
    
    dist.init_process_group(backend="nccl", init_method="env://", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    # Another place to specify device if have
    # torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

if __name__ == "__main__":
    main()
