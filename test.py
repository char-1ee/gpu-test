import torch
import torch.distributed as dist
import torch.utils.data.distributed
import argparse
import os
import sys


# Collective communication operators
class CommOp:
    def __init__(self, world_size: int):
        self.world_size = world_size

    def __call__(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    def bw_factor(self):
        raise NotImplementedError


class AllReduce(CommOp):
    def __call__(self, tensor: torch.Tensor):
        tensor = tensor.contiguous()
        dist.all_reduce(tensor)

    def bw_factor(self):
        return 2 * (self.world_size - 1) / self.world_size


class AllGather(CommOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        tensor_list = list(torch.chunk(tensor, self.world_size, dim=0))
        dist.all_gather(tensor_list, tensor)

    def bw_factor(self):
        return (self.world_size - 1) / self.world_size


class Broadcast(CommOp):
    def __call__(self, tensor: torch.Tensor, src) -> None:
        tensor.contiguous()
        dist.broadcast(tensor, src)

    def bw_factor(self):
        return 1


class AllToAll(CommOp):
    def __call__(self, tensor: torch.Tensor) -> None:
        output_tensor_list = list(torch.chunk(tensor, self.world_size, dim=0))
        input_tensor_list = output_tensor_list[:]
        dist.all_to_all(output_tensor_list, input_tensor_list)

    def bw_factor(self):
        return 1


OPS = {
    "all_reduce": AllReduce,
    "all_gather": AllGather,
    "broadcast": Broadcast,
    "all_to_all": AllToAll,
}


# Set up torch distributed environment
def dist_init():
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "49152")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )
    torch.cuda.set_device(local_rank)
    return world_size


def cleanup():
    dist.destroy_process_group()


# Timing function given tensor, operator and number of iterations
def benchmark(tensor: torch.Tensor, op: CommOp, n_iters: int) -> float:
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        op(tensor)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    average_time_ms = total_time_ms / n_iters

    return average_time_ms / 1000.0


def test_bandwidth(op_str: str, dtype=torch.float32, n_iters: int = 5):
    tensor_size = 256 * 256 * 256
    tensor = torch.rand(tensor_size, dtype=dtype, device="cuda")
    op = OPS[op_str](dist.get_world_size())

    n_warmups = 5
    for _ in range(n_warmups):
        benchmark(tensor, op, 2)

    torch.cuda.synchronize()

    average_time = benchmark(tensor, op, n_iters)
    algbw = (tensor_size * 8) / (average_time * 1e9)
    busbw = algbw * op.bw_factor()

    if dist.get_rank() == 0:
        print(f"Size (bytes): {tensor_size}")
        print(f"Data Type: {dtype}")
        print(f"Collective operators: {op_str}")
        print(f"Average Time (s): {average_time}")
        print(f"Algorithm Bandwidth (GB/s): {algbw}")
        print(f"Bus Bandwidth (GB/s): {busbw}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument(
        "--op",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_gather", "broadcast", "all_to_all"],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        print("CUDA is not available. No GPU detected on: ", device_id)
        sys.exit(1)

    if not torch.distributed.is_available():
        print(
            "Cannot found distributed packages, make sure USE_DISTRIBUTED=1 on your devices"
        )
        sys.exit(1)

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16

    dist_init()
    test_bandwidth(args.op, dtype)
    # cleanup()


if __name__ == "__main__":
    main()
