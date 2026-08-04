# B200 NBNxM CUDA Kernel Factory results

This document records the successful results and follow-up implementation
directions from Kernel Factory campaign `rvyvs8njr9317evfc04n6cscd4`, run on an
NVIDIA B200 on 2026-08-04. The campaign optimized the force-only CUDA
specialization with analytical Ewald electrostatics and geometric Lennard-Jones
combination:

`nbnxm_kernel_ElecEw_VdwLJCombGeom_F_cuda`

The extraction and packaged native baseline come from GROMACS commit
`2faf8810ffaec44b8347a1058e195e857776acb9`. The campaign sources can be
retrieved with:

```console
kf campaign kernel show <candidate-id> --source-only
kf campaign kernel traces <candidate-id> --format json
```

## Measurement contract

The packaged baseline preserves the production packed-J layout, 8x8 thread
mapping, analytical-Ewald polynomial, geometric LJ arithmetic, warp reductions,
atomics, launch bounds, and dynamic shared-memory size. It was compiled with
CUDA `-O3 --use_fast_math -std=c++17` and measured at **0.313088 ms** on B200.

The timed workload is 24 disconnected replicas of a production-generated,
post-pruning SPC/E 3000-atom snapshot:

| Axis | Value |
|---|---:|
| Padded atoms (`N`) | 82,944 |
| SCI records (`NSCI`) | 4,224 |
| Packed-J records (`NCJ`) | 45,960 |
| Exclusion records (`NEXCL`) | 12,648 |

CudaGym used 10 warm-up runs and 50 timing iterations. A single-copy workload
with 3,456 padded atoms and 176 SCI records was also evaluated for correctness.
The force comparison required every element to match under `atol=0.002`,
`rtol=0.002`, with a hard absolute-error cap of `0.02`.

## Successful candidate families

| Candidate | Latency | Speedup | Max abs. error, x24 | Max rel. error, x24 | Numerical class |
|---|---:|---:|---:|---:|---|
| `e1cf8d4e6bcf9319819e98cb4818b5e2cb73f686237460183e01cb86f223741f` | 0.155119 ms | 2.0184x | 0.00024414 | 0.007914 | Production 6/4 coefficients |
| `f55f8f8efade4cd435a6a6f0742afefe57249d5a9b160120a2e6c0f0a2f218c9` | 0.152448 ms | 2.0537x | 0.00022125 | 0.009224 | Fitted 4/4 rational |
| `70967af0dec65e72bb371fa989fa12cf4ae56db374c6498426063704220f8ba0` | 0.143247 ms | 2.1857x | 0.00118446 | 0.082863 | Fitted 3/3 rational |

The 2.0184x candidate is the primary production-math implementation target. It
uses the original GROMACS degree-6 numerator and degree-4 denominator
coefficients. Its explicit `rcp.approx.ftz.f32` denominator reciprocal matches
the reciprocal path expected from the baseline's fast-math compilation. Its
force differences come from changed work assignment and atomic accumulation
order rather than a changed Ewald fit.

The 4/4 and 3/3 candidates demonstrate the instruction cost of analytical
Ewald evaluation on this kernel. They should remain separately identifiable
from scheduling changes during attribution because they intentionally replace
the production rational approximation.

## 1. Four-way packed-J range splitting

The largest repeatedly observed scheduling improvement was decomposing one SCI
record into four independently schedulable packed-J ranges. The production
mapping launches one 64-thread CTA per SCI and lets that CTA consume the entire
`[cjPackedBegin, cjPackedEnd)` interval. The split mapping instead creates four
work units per SCI while retaining the 8x8 thread layout:

```cpp
constexpr int numChunks = 4;
const int sciIndex = workIndex >> 2;
const int chunkIndex = workIndex & 3;
```

The first production-polynomial implementation used round-robin records:

```cpp
for (int packed = begin + chunkIndex; packed < end; packed += numChunks)
```

Candidate
`9a6268bc6d6e9f8872bdb075d5bae0cf2c8603871d80a8b452d6ba489215818d`
measured **0.1898395 ms, or 1.6492x**, with this strategy and the original
GROMACS Ewald coefficients.

The mechanism is increased schedulable work and a shorter tail for uneven SCI
record lengths. The campaign profiling associated with this family reported
about 1.78 waves per SM, 35% achieved occupancy, and measurable SM/SMSP
imbalance for the one-CTA-per-SCI mapping. Four-way splitting trades duplicated
i-tile loads and additional i-force atomics for much better CTA-level load
balance.

## 2. Contiguous chunk construction

Several later candidates divided each SCI range into equal contiguous chunks:

```cpp
const int chunkSize = (end - begin + numChunks - 1) / numChunks;
const int chunkBegin = begin + chunkIndex * chunkSize;
const int chunkEnd = min(end, chunkBegin + chunkSize);
```

This retains four-way parallelism while giving each CTA sequential packed-J
access. A practical implementation should select the split count from the range
length and avoid launching empty chunks. The strongest observed parameters in
the persistent family were:

```cpp
constexpr int maxChunks = 4;
constexpr int minimumChunkLength = 3;
```

The next controlled benchmark should compare contiguous and round-robin
assignment with the same production pair arithmetic, reductions, register
limit, and dispatch policy.

## 3. Persistent chunk scheduling

Candidate
`e1cf8d4e6bcf9319819e98cb4818b5e2cb73f686237460183e01cb86f223741f`
combines the production Ewald coefficients with a device-global work queue. It
constructs up to four chunks per SCI and lets resident CTAs claim work through
an atomic counter:

```cpp
__device__ unsigned int workCounter;

if (threadIndex == 0)
{
    sharedWork = atomicAdd(&workCounter, 1U);
}
```

Each claimed work index maps to an SCI and one of its four chunks. A 256-block,
256-thread initialization kernel clears the force output and resets the queue.
The force kernel then launches exactly
`min(numSci * 4, multiprocessorCount * blocksPerSm)` resident 64-thread CTAs;
each CTA repeatedly claims work until the `numSci * 4` range is exhausted. The
kernel uses a minimum chunk length of three packed-J records so short SCI ranges
do not produce excessive fragments.

This candidate measured **0.155119 ms, or 2.0184x**, while preserving the
production Ewald rational coefficients. It is the best starting point for a
B200-oriented production prototype. The persistent queue should be introduced
as its own attribution stage after static four-way splitting so its incremental
effect is measured independently.

## 4. Direct J-atom loads

The successful kernels load each participating J atom's `float4` coordinates
and charge plus its `float2` LJ parameters directly in the consuming thread.
Campaign profiling found very low DRAM utilization for this workload and showed
that leader loads followed by six warp shuffles added dependency latency. In a
controlled campaign comparison, direct loads reduced latency from approximately
0.371 ms to 0.316 ms.

The B200 cache hierarchy can service the redundant reads efficiently, while the
direct-load path removes the shuffle dependencies from the inner packed-J loop.
This change should be evaluated after work splitting because both transformations
affect the number and locality of J loads.

## 5. Shuffle-based force reductions

The strongest J-force reduction retains the component-folding shuffle pattern
used by GROMACS. Threads first fold x, y, and z components across adjacent lanes
and then only three lanes issue scalar global atomics:

```cpp
force.x += __shfl_down_sync(fullMask, force.x, 1);
force.y += __shfl_up_sync(fullMask, force.y, 1);
force.z += __shfl_down_sync(fullMask, force.z, 1);
// Fold at offsets 2 and 4, select the component by lane.
if (iLane < 3)
{
    atomicAdd(output + 3 * jAtom + iLane, component);
}
```

For i-forces, the two warp halves reduce with offsets 8 and 16. Successful
variants either issue the resulting scalar atomics directly or stage one half
in shared memory and combine the halves before updating global memory. Direct
i-force atomics are present in the fastest persistent production-polynomial
candidate; the shared-merge variant remains useful as an isolated comparison
when porting the scheduler.

## 6. Exclusion-record-zero fast path

The packed-list ABI reserves exclusion index zero for the all-interactions
case. Successful candidates bypass the irregular exclusion-word load for this
record and synthesize the mask locally:

```cpp
const unsigned int exclusion =
        exclusionIndex == 0 ? 0xffffffffU
                            : exclusionBits[exclusionIndex * 32 + lane];
```

This removes an indexed load from the inner loop for the common no-exclusion
path. It is independent of the SCI scheduler and can be added as a separate
attribution stage after the persistent production-polynomial kernel is
reproduced.

## 7. Register and launch configuration

The successful production-style kernels use 64-thread `dim3(8, 8, 1)` CTAs and
preserve the 64-register working set associated with eight resident CTAs per SM:

```cpp
__global__ __launch_bounds__(64, 16)
void nbnxmKernel(...);
```

The implementation should be compiled with ptxas resource reporting at every
attribution stage. Record registers per thread, static and dynamic shared
memory, stack frame size, and spills alongside latency. Scheduling experiments
should retain the same compiler resource target so occupancy and arithmetic
changes are not conflated.

## Analytical-Ewald instruction variants

The campaign produced two shorter analytical-Ewald approximations that compose
with the scheduling work but must be measured as explicit numerical modes.

### Degree-4/4 rational

Candidate
`f55f8f8efade4cd435a6a6f0742afefe57249d5a9b160120a2e6c0f0a2f218c9`
uses a Horner-evaluated minimax 4/4 rational over `z^2` in `[0, 9.9]`. It
measured **0.152448 ms, or 2.0537x**, with maximum x24 force errors of
`0.00022125` absolute and `0.009224` relative.

### Degree-3/3 rational

Candidate
`70967af0dec65e72bb371fa989fa12cf4ae56db374c6498426063704220f8ba0`
uses a six-FMA degree-3/3 rational together with static four-way packed-range
splitting. It measured **0.143247 ms, or 2.1857x**, with maximum x24 force
errors of `0.00118446` absolute and `0.082863` relative.

These variants quantify the latency available from shortening the hottest
scalar arithmetic path. They should be maintained as separate optional stages
after the production-coefficient scheduling result has been reproduced.

## Recommended attribution ladder

Use the same B200, workload set, compiler flags, and timing configuration for
every stage. Add one logical change per stage:

| Stage | Change from previous stage | Primary measurement |
|---|---|---|
| A | Packaged production baseline | Reproduce 0.313088 ms and compiler resources |
| B | Static two-way packed-J split | Establish the split-count curve |
| C | Static four-way packed-J split | Reproduce the production-polynomial split result |
| D | Contiguous chunks with a minimum length | Isolate chunk-construction policy |
| E | Persistent atomic work queue | Isolate dynamic load balancing |
| F | Direct per-thread J loads | Measure dependency removal and load behavior |
| G | Exclusion-index-zero fast path | Measure inner-loop indexed-load removal |
| H | Direct versus shared-merged i-force updates | Select the reduction policy |
| I | Optional 4/4 Ewald rational | Measure the higher-accuracy shortened polynomial |
| J | Optional 3/3 Ewald rational | Measure the minimum-depth fitted polynomial |

For every stage, retain per-workload force errors, kernel latency, register
count, shared-memory use, stack size, spills, and the exact CUDA source diff.

## Source and artifact locations

- Campaign configuration: `artifacts/workspace-b200/campaign.yaml`
- Definition and reference: `artifacts/workspace-b200/definition.json`
- Workload matrix: `artifacts/workspace-b200/workload.jsonl`
- Native packaged baseline: `artifacts/workspace-b200/baseline_solution.json`
- Prepared baseline timing: `artifacts/workspace-b200/baselines.jsonl`
- Captured production snapshot: `artifacts/snapshot-spce3000-post-prune/`

All paths above are relative to
`benchmarks/kernel-factory/nbnxm_cuda_force/`.
