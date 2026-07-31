# CUDA + MPI Domain Decomposition (2D Diffusion)

[![CUDA](https://img.shields.io/badge/CUDA-C%2B%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![MPI](https://img.shields.io/badge/MPI-multi--process-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multi-process GPU PDE**: vertical domain decomposition of 2D diffusion with **MPI**, per-rank CUDA stencils, and **halo exchange**. Demonstrates the hybrid model used on modern supercomputers (MPI between nodes/GPUs + CUDA on device).

---

## Skills demonstrated

| Skill | Detail |
|-------|--------|
| SPMD MPI | ranks, `MPI_Init`, communicators |
| Domain decomposition | split `ny` across ranks |
| Halo / ghost exchange | neighbor boundary rows |
| CUDA kernels per rank | 2D diffusion stencil |
| Optional GPU-aware / RDMA MPI | env flag checks in code |
| Batch job scripts | `scripts/job.sbatch` (cluster style) |

**Resume bullets:**
- Implemented **MPI + CUDA** 2D diffusion with slab decomposition and halo exchange.
- Structured a hybrid parallel application suitable for multi-GPU / multi-node HPC systems.

---

## Build & run

```bash
# Needs MPI + CUDA
make ARCH=sm_80
mpirun -np 4 ./bin/diffusion2d_mpi 10 100
```

On clusters, see `scripts/job.sbatch`.

---

## Attribution

Based on CSCS–USI Summer School CUDA–MPI diffusion exercise; portfolio packaging by Innocent Kisoka.
