# Gloo / MPI multi-thread stress

Date: 2026-07-28T14:05:00Z

| Run | Status | passed | failed | skipped | log |
|-----|--------|--------|--------|---------|-----|
| single | PASS | 17 | 0 | 0 | `samples/out/gloo_mpi_stress/single_mt.log` |
| mpi-mpirun(n=2) | PASS | 10 | 0 | 0 | `samples/out/gloo_mpi_stress/mpi_mpirun.log` |

## Mac conclusions (verified)

| Test | Result |
|------|--------|
| T1 Gloo forceCollective ws=1 | PASS |
| T2 8-thread Gloo PGs | PASS (~64 coll/s) |
| T3 4-thread local DDP | PASS |
| T4 MPI create+allreduce ws=1 | PASS |
| T5 Gloo latency | p50 ≈ 0.02–0.03 ms |
| T6 MPI 4-thread single PG | PASS (~40k coll/s) |
| T7 wrapper BackendType.MPI | PASS |
| T8 MPI latency | p50 ≈ 0.01 ms |
| mpirun -n 2 direct allreduce | 1+2→3.0 PASS |
| mpirun -n 2 multi-thread coll | 4/4 PASS both ranks |
| mpirun -n 2 wrapper MPI | PASS |

## Fixes applied this session
1. `ProcessGroupMPI` collective `@IntrusivePtr("c10d::Work")` + `c10d::` nullValues
2. `torch_mpi` LoadEnabled: do not clear `platform.library` when packaged `jnitorch_mpi` exists
3. Rebuild `libjnitorch_mpi.dylib` + package into platform jar (MPI libtorch)
4. `scripts/run_gloo_mpi_stress.sh` zsh-safe CP + mpirun wrapper
5. `scripts/build_jnitorch_mpi.sh` openblas linkpath
