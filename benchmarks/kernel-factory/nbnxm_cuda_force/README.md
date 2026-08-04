# GROMACS CUDA NBNxM Kernel Factory campaign

This campaign targets the steady-state force-only CUDA specialization with
analytical Ewald electrostatics and geometric Lennard-Jones combination:

`nbnxm_kernel_ElecEw_VdwLJCombGeom_F_cuda`

The B200 campaign results, successful implementation families, candidate IDs,
and recommended attribution ladder are recorded in
[`CAMPAIGN_RESULTS_B200.md`](CAMPAIGN_RESULTS_B200.md).

The correctness reference is independent PyTorch. The measured baseline is the
existing GROMACS CUDA algorithm packaged as `baseline.solution.json`, including
the production packed-J layout, 8x8x8 work mapping, analytical-Ewald polynomial,
warp reductions, atomics, launch bounds, and dynamic shared-memory launch.

## Generate the campaign input

Configure CUDA explicitly for the detected host architecture. For the local Ada
GPU, use SM89:

```console
cmake -S . -B /tmp/gromacs-kf-build -G 'Unix Makefiles' \
    -DGMX_GPU=CUDA \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DGMX_CUDA_ARCHITECTURES=89 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGMX_BUILD_OWN_FFTW=OFF \
    -DGMX_FFT_LIBRARY=fftw3 \
    -DBUILD_TESTING=ON
cmake --build /tmp/gromacs-kf-build --target gmx -j 8
```

Capture a production-generated list and package it:

```console
/tmp/gromacs-kf-build/bin/gmx nonbonded-benchmark \
    -gpu -gpu-prune -device-id 0 -size 1 -iter 10 -time \
    -kernel-factory-snapshot benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/snapshot-spce3000-post-prune

python benchmarks/kernel-factory/nbnxm_cuda_force/pack_snapshot.py \
    benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/snapshot-spce3000-post-prune \
    --min-blocks 4096 \
    --target-hardware B200 \
    --out benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200
```

The packer emits all launch inputs:

- `definition.json` with the independent, memory-bounded reference;
- `workload.jsonl` with a small correctness-only case and a saturated full case;
- safetensors blobs retaining the raw production packed-J ABI;
- `baseline.solution.json` containing the runnable native CUDA baseline;
- `audit.json` recording source provenance, target, replication, and block count.

The default saturated workload replicates disconnected captured systems until it
has at least 4,096 independent CUDA blocks. For the current snapshot that is 24
copies, 82,944 padded atoms, 4,224 SCI blocks, 45,960 packed-J records, and a
roughly 5 MiB input blob. Replication adjusts every atom, cluster, packed-list,
and exclusion index; it does not introduce cross-copy interactions.

## Validate locally

```console
python benchmarks/kernel-factory/nbnxm_cuda_force/validate_snapshot.py \
    benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/snapshot-spce3000-post-prune \
    benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200 \
    --device cuda
```

`validate_baseline.py` can build the torch extension when Ninja is installed, or
accept a prebuilt extension through `--module`. Always set `TORCH_CUDA_ARCH_LIST`
explicitly before its build path.

## Prepare and start

After `kf auth login`:

```console
kf campaign init gromacs-nbnxm-cuda-force \
    --definition benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200/definition.json \
    --workloads benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200/workload.jsonl \
    --baseline-solution benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200/baseline.solution.json \
    --asset benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200/nbnxm_spce3000_x1.safetensors \
    --asset benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/campaign-b200/nbnxm_spce3000_x24.safetensors \
    --prompt-file benchmarks/kernel-factory/nbnxm_cuda_force/prompt.md \
    --gpu-spec b200 \
    --language cuda_cpp_only \
    --effort high \
    --out benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/workspace-b200

kf campaign prepare --from benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/workspace-b200/campaign.yaml
kf --format json campaign start \
    --from benchmarks/kernel-factory/nbnxm_cuda_force/artifacts/workspace-b200/campaign.yaml \
    --watch
```

Inspect the generated `baselines.jsonl` after `prepare` and before starting. It
must contain a native-baseline timing for every `full` workload; a synthesized
reference timing is not an acceptable baseline for this campaign.
