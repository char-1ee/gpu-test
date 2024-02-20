This script tests GPU communication bandwidth utilizing NCCL primitives. It supports operations like all-reduce, all-to-all, all-gather, and broadcast across both single-node-multi-card and multi-node-multi-card environments. The implementation relies solely on Pytorch and Python's built-in libraries.

RUNNING THE SCRIPT

By default, the master address is set to 10.20.1.85.

To run the script, follow these steps:

1. Navigate to the gpu-test directory:
   `cd gpu-test`

2. Execute the run script:
   `bash run.sh`
