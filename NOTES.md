# Team Notes — Greedy Fusion Solver

## Problem TL;DR

Given a DAG of ops (MatMul/Pointwise) over tensors, schedule them into **subgraphs** with execution **granularity** `[w,h,k]` to minimize total latency under a 3-tier memory model:

- **Slow mem**: ∞ capacity, limited bandwidth — all data starts/ends here
- **Fast mem**: finite capacity, free access — working scratchpad
- **Ephemeral**: zero cost — intermediates within a fused subgraph

Per-tile latency uses a **roofline model**: `max(compute, mem_in + mem_out)`.

## Code Walkthrough (`solver.cpp`)

### 1. JSON I/O (lines 22–174)

Hand-rolled recursive descent parser for the input format. `JVal` supports object/array/number/string access. `read_problem()` also builds derived fields:
- `producer[t]` — which op produces tensor `t` (-1 = graph input)
- `consumers[t]` — which ops consume tensor `t`
- `graph_ins` / `graph_outs` — tensors with no producer / no consumer

### 2. Subgraph Analysis (`analyze`, lines 236–264)

Given a set of ops, classifies every tensor into:

| Category | Definition | Memory cost |
|---|---|---|
| **input boundary** | consumed but not produced in subgraph | load from slow mem |
| **output boundary** | produced AND (consumed outside OR graph output) | evict to slow mem |
| **ephemeral** | produced AND consumed only within subgraph | **zero** |

This is the key insight for fusion: merging producer→consumer makes the intermediate tensor ephemeral, eliminating its memory transfer cost.

### 3. Slice Sizes (lines 189–227)

Two functions, both take the **max** across all consuming ops when a tensor has multiple roles:

| Function | Purpose | MatMul LHS | MatMul RHS | Pointwise |
|---|---|---|---|---|
| `input_slice` | Working-set check (instantaneous) | `h × k` | `w × k` | `w × h` |
| `tile_mem_in` | Latency calc (total per tile) | `h × K_full` | `w × K_full` | `w × h` |

The distinction matters for split-K: working set uses the small per-step slice (`k`), but total memory transferred per tile uses the full reduction dimension (`K`).

### 4. Latency Model (`calc_latency`, lines 280–307)

Per spatial tile, simplified roofline (no traversal reuse, raster order):

```
tiles       = ceil(out_W / w) × ceil(out_H / h)
nat_scale   = ceil(w / native_w) × ceil(h / native_h)    // padding penalty
compute     = Σ base_cost × nat_scale
mem_in      = Σ tile_mem_in(boundary inputs) / bandwidth
mem_out     = Σ (w × h) per boundary output / bandwidth
tile_latency = max(compute, mem_in + mem_out)
total        = tiles × tile_latency
```

**Key property**: total memory is independent of `k` (split-K doesn't change total bytes, only instantaneous working set).

### 5. Granularity Search (`find_best_gran`, lines 321–358)

Enumerates power-of-2 candidates for `[w, h, k]`:
- Checks `working_set ≤ fast_memory_capacity`
- Picks lowest latency, preferring larger tiles on ties
- For `k`: searches all powers of 2 up to `max_K` across MatMul ops

### 6. Greedy Fusion (lines 471–535)

```
1. Init: each op = its own subgraph
2. Loop:
   a. Find all adjacent subgraph pairs (connected via tensor edge)
   b. For each pair, check cycle safety (BFS: no indirect path a→...→b)
   c. Evaluate merged latency, compute benefit = lat_before - lat_after
   d. Execute the highest-benefit merge
   e. Repeat until no beneficial merge exists
```

**Cycle check** (lines 416–451): BFS from `sg_a`'s successors (excluding `sg_b`) — if `sg_b` is reachable via other subgraphs, merging would create a dependency cycle.

## Integration Points for Teammates

| Module | Current state | Where to plug in |
|---|---|---|
| **Split-K** | `k` searched in `find_best_gran`; latency uses simplified total model | Replace `calc_latency` with per-step roofline: `Σ max(compute_step, mem_step)` |
| **Retention** | `tensors_to_retain = []` always | After fusion, decide which output tensors to keep in fast mem; adjust `mem_in`/`mem_out` in latency calc |
| **Traversal** | `traversal_orders = null` always | Add zig-zag/snake tile ordering; adjust per-tile `mem_in` for data reuse (see PROBLEM.md Example 4) |

## Verification Strategy

**The organizers do NOT provide expected outputs.** The `Evaluate()` function in `mlsys.h` depends on Google-internal `absl` and isn't runnable standalone.

### How to verify correctness

1. **Hand-check against PROBLEM.md examples** — The 5 worked examples have exact latency numbers. We verified our solver matches Example 1B (3276.8 for the fused case).

2. **Build our own evaluator** — see `verify.cpp` below. It checks:
   - Every op appears in exactly one subgraph
   - Subgraphs are in valid topological order
   - Working set fits in `fast_memory_capacity` per tile
   - Reported `subgraph_latencies` match our recomputation
   - All graph outputs are eventually evicted to slow memory

3. **Lower/upper bound reasoning**:
   - **Lower bound** = `max(total_compute, total_memory) / num_tiles` — unreachable but useful sanity check
   - **Upper bound** = unfused baseline (each op alone) — our fusion should always beat this
   - **Naive baseline** = `Σ max(base_cost, tensor_size/bw)` per op

4. **Comparative testing** — run multiple strategies and compare:
   - All-unfused baseline
   - Our greedy fusion
   - Brute-force (small benchmarks only: enumerate all possible subgraph partitions)

5. **Invariant checks** (things that must always hold):
   - `Σ subgraph_latencies == total_latency`
   - Every tensor consumed outside a subgraph must be in `out_bd` (evicted or retained)
   - No ephemeral tensor is accessed by ops in different subgraphs

### What we can't verify

- The organizers' exact `Evaluate()` implementation may use a more detailed per-step roofline model (sum of per-step maxima instead of max of totals). Our simplified model is a **lower bound** on the true latency. The reported `subgraph_latencies` should ideally use the detailed model for accuracy.
- Traversal order effects on data reuse are not modeled — our latencies assume worst-case raster order.

## Current Results

| Benchmark | Ops | Subgraphs | Latency | Unfused Baseline | Fusion Speedup | Time |
|---|---|---|---|---|---|---|
| example | 2 | 1 | 3,277 | 6,554 | 2.00× | <1ms |
| mlsys-2026-1 | 5 | 2 | 183,501 | 340,787 | 1.86× | <1ms |
| mlsys-2026-5 | 19 | 8 | 845,278 | 1,083,051 | 1.28× | 10ms |
| mlsys-2026-9 | 32 | 24 | 20.0M | 22.3M | 1.11× | 20ms |
| mlsys-2026-13 | 63 | 21 | 11.5M | 11.8M | 1.03× | 400ms |
| mlsys-2026-17 | 103 | 73 | 5.0M | 5.1M | 1.02× | 250ms |

**Note:** Low speedup on benchmarks 13/17 indicates they're heavily memory-bound with limited fusion opportunities — retention and traversal optimizations should help most here.
