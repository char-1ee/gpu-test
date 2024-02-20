#!/bin/bash

# NCCL_DEBUG=INFO torchrun --nnodes=1 --nproc-per-node=8 --master_addr=10.20.1.85 test.py 
# torchrun --nnodes=2 --nproc-per-node=8 --master_addr=10.20.1.85 test.py 

date 

TRAINERS_PER_NODE=8 
NUM_ALLOWED_FAILURES=1
SCRIPT="test.py"
LOG_DIR="./logs"
MASTER_ADDR=10.20.1.85

run_torchrun() {
  local num_nodes=$1
  echo "Running brandwidth tests on ${num_nodes} node(s)..."
  torchrun \
    --nnodes=${num_nodes} \
    --nproc-per-node=${TRAINERS_PER_NODE} \
    --max-restarts=${NUM_ALLOWED_FAILURES} \
    --master_addr=${MASTER_ADDR} \
    ${SCRIPT}
}

mkdir -p ${LOG_DIR}
node_counts=(1 2 4 8)
for nodes in "${node_counts[@]}"; do
  run_torchrun $nodes > "${LOG_DIR}/torchrun_${nodes}_nodes.log" 2>&1
done

echo "Brandwidth tests end"

