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

| Benchmark | Ops | SGs | Ephemerals | Latency | Unfused | Speedup |
|---|---|---|---|---|---|---|
| example | 2 | 1 | 1 | 3,277 | 6,554 | 2.00× |
| mlsys-2026-1 | 5 | 2 | 3 | 148,344 | 340,787 | 2.30× |
| mlsys-2026-5 | 19 | 7 | 12 | 690,221 | 1,083,051 | 1.57× |
| mlsys-2026-9 | 32 | 24 | 8 | 16.7M | 22.3M | 1.34× |
| mlsys-2026-13 | 63 | 21 | 42 | 11.4M | 11.8M | 1.03× |
| mlsys-2026-17 | 103 | 17 | 86 | 5.0M | 5.1M | 1.02× |

**Geometric mean speedup: ~1.47×**

### Optimization breakdown

#### 1. Greedy fusion (Phase 1 — positive-benefit merges)
The baseline: every op in its own subgraph. Greedy loop finds adjacent pairs whose merge produces `benefit = lat_before − lat_after > 0`, picks the best, and repeats. The benefit comes from making intermediate tensors **ephemeral** (no HBM round-trip). For example, fusing Op0 (MatMul) + Op1 (Pointwise) makes Op0's output ephemeral, saving `2 × tensor_size / bw` per tile.

Results: B-1 fuses 4/5 ops into one SG (3 ephemerals), B-5 fuses MatMul+Pointwise pairs. B-9/13/17 achieve limited fusion because the large weight tensors (4096×4096 in B-13, 2048×128 in B-17) make merged working sets exceed fast memory capacity.

#### 2. Zig-zag traversal (`assign_traversals` in solver.cpp)
For MatMul subgraphs with >1 spatial tile, we generate a snake-order tile sequence. The data reuse pattern:
- **Same row (ty unchanged):** LHS strip (h×K) reused → skip LHS load
- **Row transition (tx unchanged due to zig-zag):** RHS strip (w×K) reused → skip RHS load

Per-tile latency model: `calc_latency_final` walks the tile grid, tracking `prev_tx/prev_ty`. For each tile, only loads strips that changed from the previous tile.

**Concrete B-9 trace (SG[1], Op2 MatMul, gran [512,256,128], 2×4=8 tiles):**
```
LHS = h×K/bw = 256×4096/25 = 41,943  (depends on ty)
RHS = w×K/bw = 512×4096/25 = 83,886  (depends on tx)
Out = w×h/bw = 512×256/25  = 5,243
Compute = 5000 × nat_scale(8) = 40,000

Raster (all 8 tiles load both): 8 × max(40K, 41.9K+83.9K+5.2K) = 8 × 131,072 = 1,048,576
Zig-zag:
  Tile (0,0): LHS+RHS → max(40K, 131K) = 131,072
  Tile (1,0): LHS reused → max(40K, 89.1K) = 89,129
  Tile (1,1): RHS reused → max(40K, 47.2K) = 47,186
  Tile (0,1): LHS reused → max(40K, 89.1K) = 89,129
  ...pattern repeats for rows 2-3
  Total = 131,072 + 3×89,129 + 3×47,186 + 89,129 = 629,146 (40% savings)
```

**Why it doesn't help B-13/17:**
- B-13: all SGs have 1×1 tiles (gran [4096,128,8] matches output 4096×128). No multi-tile grid.
- B-17: multi-tile SGs are compute-bound (compute 80K >> memory 2.8K). Zig-zag saves memory but `max(compute, mem)` still equals compute.

#### 3. Zero-benefit fusion (Phase 2 — ephemeral-creating merges)
After Phase 1, a second loop merges pairs where `benefit ≥ 0` (no latency increase) AND the merge creates new ephemeral tensors. This is valuable for **compute-bound** graphs where our roofline `max(compute, mem)` = `compute` regardless of memory savings, but the evaluator's actual hardware still benefits from less HBM traffic.

**Concrete B-17 result:** The graph has 8 multi-head attention blocks (5 MatMul ops each: 3 projections + 2 attention matmuls) plus up/down projections and pointwise ops. Phase 1 only merged MatMul+Pointwise pairs (12 SGs of 2 ops). Phase 2 fused:
- Each 5-op attention head chain into 1 SG (8 SGs × 5 ops = 40 ops, 32 new ephemerals)
- Remaining 47 ops (projections, pointwise, etc.) into 1 mega-SG (46 new ephemerals)
- **73 → 17 subgraphs, 86 total ephemeral tensors, zero latency change**

#### 4. Retention (`assign_retention` in solver.cpp)
After toposort, for consecutive pairs (SG_i, SG_{i+1}), we check if any output tensor of SG_i is an input to SG_{i+1}. If retaining the full tensor fits in both SGs' capacity budgets, we retain it. Capacity checks:
- **Producer:** `working_set(SG_i) + (T_full − T_tile_out) ≤ fast_cap` (accumulate full tensor instead of evicting tiles)
- **Consumer:** `working_set(SG_{i+1}) − T_slice_in + T_full ≤ fast_cap` (full tensor resident instead of loading slices)

**B-13 SG[18]→SG[19]:** Tensor 42 (128×128 = 16,384) retained. Producer ws + extra = 21,299 + 0 ≤ 600K ✓. Consumer ws − slice + full = trivially fits ✓. Saves `2 × 16,384/50 = 655` latency units (evict + reload). Small but correct.

**Why retention is rare:** Most intermediate tensors are large (512×512 = 262K on B-1, 1024×1024 = 1M on B-9, 4096×128 = 524K on B-13). These exceed or nearly fill fast memory capacity, leaving no room alongside the next SG's working set.

### Bottleneck analysis

#### B-13 (1.03× speedup) — memory-bound, capacity-limited
**Architecture:** 16 parallel chains of 2 chained MatMuls each, feeding into a pointwise reduction network. Each chain: `T_in(4096×128) @ T_weight(4096×4096) → ephemeral → @ T_weight2(4096×4096) → T_out(4096×128)`.

**Why it's stuck at 1.03×:**
- Gran [4096,128,8] gives 1 spatial tile (output = 4096×128). Zig-zag irrelevant.
- K=4096 with k=8 → 512 k-steps. Working set per step: `LHS(128×8) + RHS0(4096×8) + RHS1(4096×8) + Acc(4096×128) = 590,848 / 600,000 cap`. Only 9K spare.
- **Can't increase k:** k=16 → ws = 657K > 600K. Doesn't fit.
- **Can't reduce w for more tiles:** w=2048 fits and allows k=64 (only 128 k-steps), but now 2 tiles load the 4096×4096 RHS twice. Latency = 702K > 692K. Worse.
- **Fusion limited:** The 16 chains are parallel (no shared intermediates). Only the final pointwise network benefits from fusion (26 ops → 1 SG, already fused).

**Tested alternative granularities:**
| Granularity | WS | Tiles | k-steps | Latency | vs current |
|---|---|---|---|---|---|
| [4096, 128, 8] | 590K | 1 | 512 | 692,060 | **baseline** |
| [4096, 128, 16] | 657K | — | — | OOM | ✗ |
| [2048, 128, 64] | 398K | 2 | 64 | 702,546 | +1.5% worse |
| [4096, 64, 8] | 328K | 2 | 512 | 1,001,574 (zz) | +45% worse |

#### B-17 (1.02× speedup) — fully compute-bound
**Architecture:** 8 multi-head attention blocks (128×128 intermediate tensors), 8 up/down projection pairs (128×2048 outputs), pointwise reduction network.

**Why it's stuck at 1.02×:**
- 128×128 MatMul SGs: compute = 10,000 × nat_scale(1) = 10,000. Memory = 2 × 16,384/100 = 327.68. Ratio: **30:1 compute-dominant**.
- 128×2048 MatMul SGs: compute = 10,000 × nat_scale(8) = 80,000. Memory = (131K + 16K)/100 + 13K/100 = 1,475 + 1,311 = 2,786. Ratio: **28:1 compute-dominant**.
- In roofline model, `max(compute, mem) = compute` regardless of memory savings. Fusion/zig-zag/retention change `mem` but not `max`.
- Total compute per op is fixed: `base_cost × ceil(W/nat_w) × ceil(H/nat_h)`, independent of tile size. No granularity change helps.

#### B-9 (1.34× speedup) — memory-bound, limited by skip connections
**Architecture:** 8 transformer-like blocks, each: MatMul(up-proj) + Pointwise(activation) → MatMul(down-proj) → Pointwise(residual add + skip connection). Skip tensors: T20, T24, T28, T32, T36, T40, T44 (each 1024×1024 = 1M).

**Fusion limits:** Op3 (residual add producing T20) takes T19 (from down-proj) + T0 (skip input). T20 is consumed by both Op4 (next block's up-proj, fused into SG[3]) and Op7 (next block's residual add, in SG[5]). T20 cannot be ephemeral unless both consumers are in the same SG as Op3.

**Why full chain fusion fails:** Fusing {Op3,Op4,Op5,Op6,Op7} would make T20 ephemeral, but at gran [256,128,64]: WS = 131K < 250K ✓. However tiles = 4×8 = 32. The huge down-proj RHS (T4: 1024×4096) costs `tile_mem_in = 256×4096/25 = 41,943` per tile. Total: 32 × 65,536 = 2.1M vs current 755K for the 4 separate SGs. **2.8× worse.**

**Recomputation analysis:** Include Op0-Op3 in SG_B alongside Op6-Op7 so T20 is recomputed:
- Recompute cost: (Op0=5K + Op1=200 + Op2=5K + Op3=500) × nat_scale(8) = 85,600 per tile
- HBM savings: T20 evict + load = 2 × 1M/25 = 83,886 total (not per tile)
- Compute overhead (85,600/tile × ntiles) >> savings (83,886 total). **Not beneficial.**

**Retention analysis:** T20 (1,048,576 elements) > fast_cap (250,000). Cannot be retained in fast memory at all.

#### Per-step split-K investigation (all benchmarks)
Tested whether modeling each k-step individually (instead of per-tile totals) changes results. For B-13's typical SG (K=4096, k=8, 512 steps):
```
Per-step: compute = 625, mem_in = 1,331.2 → every step mem-bound
Sum = 511 × 1,331.2 + (1,331.2 + 10,485.8) = 692,060
Simplified: max(320,000, 681,574 + 10,486) = 692,060
Identical.
```

The per-step model only diverges when some steps are compute-bound and others memory-bound (e.g., full-loading an input in step 1, making steps 2+ compute-bound as in PROBLEM.md Example 5). This requires an input that fits fully resident alongside other data. On all benchmarks, no such input fits:
- B-13: full LHS (524K) + streaming + acc = 1.1M >> 600K cap
- B-9: full LHS (1M) >> 250K cap
- B-17: already compute-bound, no split-K benefit possible

### Timeline
1. **Greedy fusion + granularity search + JSON I/O** → solver.cpp, verify.cpp, Makefile
2. **Zig-zag traversal** → `gen_zigzag`, `matmul_role`, `calc_latency_final`, `assign_traversals`
3. **Retention** → `assign_retention`, updated `write_solution` with `retained_in` tracking
4. **Phase 2 fusion** → second loop in `greedy_fusion` accepting benefit=0 when ephemerals created
5. **Investigated & rejected:** per-step split-K (identical on all benchmarks), multi-hop retention (tensors too large or savings negligible), recomputation (compute overhead > HBM savings), alternative granularities for B-13/B-17 (all worse or OOM)
