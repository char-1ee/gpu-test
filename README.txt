GPU Communication Bandwidth Test
================================

This suite of scripts tests GPU communication bandwidth utilizing NCCL (NVIDIA Collective Communications Library) primitives. It supports operations such as all-reduce, all-to-all, all-gather, and broadcast across configurations including single-node-multi-card and multi-node-multi-card environments. The implementation relies exclusively on PyTorch and Python's built-in libraries.

For detailed approaches on NCCL bandwidth computation, see the NCCL Tests Performance Documentation at:
https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md


Prerequisites
-------------
- PyTorch
- Python's built-in libraries
- Access to one or more nodes equipped with NVIDIA GPUs


Running the Tests
-----------------

Single Node:
By default, the master address is set to 10.20.1.85.

To run the test on a single node, follow these steps:

1. Prepare the Environment: Navigate to root working directory.
   cd gpu-test

2. Run the Script: Execute the provided bash script.
   bash run.sh

Multiple Nodes:
For scenarios involving multiple nodes, need to manually execute torchrun on all nodes with the appropriate configurations.

- Master Node (Address: 10.20.1.83):
  `torchrun --nnodes=2 --nproc-per-node=8 --node_rank=0 --master_addr=10.20.1.83 test.py`

- Slave Node (Address: 10.20.1.179):
  `torchrun --nnodes=2 --nproc-per-node=8 --node_rank=1 --master_addr=10.20.1.83 test.py`

Ensure the `--nnodes`, `--nproc-per-node`, `--node_rank`, and `--master_addr` arguments are correctly set for specific multi-node setup.

