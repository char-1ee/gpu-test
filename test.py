import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import os
import time

def setup(rank, world_size):
    # TODO: config addr and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def test_bandwidth(rank, world_size, dtype, op):
    setup(rank, world_size)

    # Create a tensor
    tensor_size = 1024 * 1024 * 256  # 256 MB
    tensor = torch.rand(tensor_size, dtype=dtype, device='cuda')

    # Warm-up
    for _ in range(10):
        if op == 'all-reduce':
            dist.all_reduce(tensor)
        elif op == 'all-gather':
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
        elif op == 'broadcast':
            dist.broadcast(tensor, src=0)
        elif op == 'all-to-all':
            tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_to_all(tensors, [tensor]*world_size)

    # Benchmark
    start_time = time.time()
    for _ in range(50):
        if op == 'all-reduce':
            dist.all_reduce(tensor)
        elif op == 'all-gather':
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
        elif op == 'broadcast':
            dist.broadcast(tensor, src=0)
        elif op == 'all-to-all':
            tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_to_all(tensors, [tensor]*world_size)

    duration = time.time() - start_time
    bandwidth = (tensor.nelement() * tensor.element_size() * 50) / duration / 1e9

    if rank == 0:
        print(f'Operation: {op}, Data type: {dtype}, Bandwidth (GB/s): {bandwidth}')

    cleanup()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Bandwidth Test')
    parser.add_argument('--world-size', type=int, default=4, help='number of GPUs to use')
    parser.add_argument('--dtype', default='float32', choices=['float32', 'float16'], help='data type')
    parser.add_argument('--op', default='all-reduce', choices=['all-reduce', 'all-gather', 'broadcast', 'all-to-all'], help='operation to perform')

    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == 'float32' else torch.float16
    op = args.op

    mp.spawn(test_bandwidth,
             args=(args.world_size, dtype, op),
             nprocs=args.world_size,
             join=True)

if __name__ == "__main__":
    main()

