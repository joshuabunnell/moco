<!-- source: https://docs.rc.asu.edu/supercomputer-hardware -->
# Supercomputer Hardware | ASU RC Docs

## Sol Specs

Sol is "a homogeneous supercomputer" where "processors and interconnects that are of the same type, brand, and architecture" create uniformity that "simplifies system management and optimization."

### Sol Node Types and Specifications

| Node Type | CPU | Memory | Accelerator |
|-----------|-----|--------|-------------|
| Standard Compute | 128 Cores (2x AMD EPYC 7713 Zen3) | 512 GiB | N/A |
| High Memory | 128 Cores (2x AMD EPYC 7713 Zen3) | 2048 GiB | N/A |
| GPU A100 | 48 Cores (2x AMD EPYC 7413 Zen3) | 512 GiB | 4x NVIDIA A100 80GiB |
| GPU A30 | 48 Cores (2x AMD EPYC 7413 Zen3) | 512 GiB | 3x NVIDIA A30 24GiB |
| GPU MIG | 48 Cores (2x AMD EPYC 7413 Zen3) | 512 GiB | 16x NVIDIA A100 sliced into 20GiB and 10GiB |
| Xilinx FPGA | 48 Cores (2x AMD EPYC 7443 Zen3) | 256 GiB | 1x Xilinx U280 |
| Bitaware FPGA | 52 Cores (Intel Xeon Gold 6230R) | 376 GiB | 1x BittWare 520N-MX |
| NEC FPGA | 48 Cores (2x AMD EPYC 9274F Zen4) | 512 GiB | 1x NEC Vector Engine |
| GraceHopper | 72 Cores (NVIDIA Grace CPU aarch64) | 512 GiB | 1x NVIDIA GH200 480GB |
| GPU MI200 | 24 Cores (AMD EPYC 9254) | 77 GiB | 2x AMD MI200 |

## Phoenix Specs

Phoenix is "a heterogeneous supercomputer" featuring "processors and interconnects that are of different types, brands, and architectures." This diversity "can complicate system management and optimization but offers a wider range of available hardware."

### Phoenix Node Types and Specifications

| Node Type | CPU | Memory | Accelerator |
|-----------|-----|--------|-------------|
| Standard Compute | 28 Cores (2x Intel Broadwell) | 128 GiB | N/A |
| High Memory | 56 Cores (2x Intel Skylake Xeon Gold 6132 @ 2.6GHz) | 1500 GiB | N/A |
| GPU V100 | 40 Cores (2x Intel Skylake Xeon Gold 6148 @ 2.40GHz) | 360 GiB | 4x NVIDIA V100 32GiB |
| Intel Phi | 256 Cores (2x Intel Knights Landing Phi) | 128 GiB | N/A |
