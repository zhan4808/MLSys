#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

// ============================================================
// Minimal JSON parser (handles the specific input format)
// ============================================================

struct JVal {
    enum T { NUL, NUM, STR, ARR, OBJ } t = NUL;
    double n = 0;
    string s;
    vector<JVal> a;
    vector<pair<string, JVal>> o;

    const JVal& operator[](const char* key) const {
        for (auto& [k, v] : o)
            if (k == key) return v;
        static JVal nil;
        return nil;
    }
    const JVal& operator[](size_t i) const { return a[i]; }
    int sz() const { return t == ARR ? (int)a.size() : (int)o.size(); }
    int64_t i64() const { return (int64_t)n; }
};

JVal jparse(const string& s, size_t& p) {
    auto ws = [&] { while (p < s.size() && isspace(s[p])) p++; };
    ws();
    if (p >= s.size()) return {};
    if (s[p] == '{') {
        JVal v;
        v.t = JVal::OBJ;
        p++;
        ws();
        while (p < s.size() && s[p] != '}') {
            auto k = jparse(s, p);
            ws();
            if (p < s.size() && s[p] == ':') p++;
            auto val = jparse(s, p);
            v.o.push_back({k.s, val});
            ws();
            if (p < s.size() && s[p] == ',') p++;
        }
        if (p < s.size()) p++;
        return v;
    }
    if (s[p] == '[') {
        JVal v;
        v.t = JVal::ARR;
        p++;
        ws();
        while (p < s.size() && s[p] != ']') {
            v.a.push_back(jparse(s, p));
            ws();
            if (p < s.size() && s[p] == ',') p++;
        }
        if (p < s.size()) p++;
        return v;
    }
    if (s[p] == '"') {
        JVal v;
        v.t = JVal::STR;
        p++;
        while (p < s.size() && s[p] != '"') v.s += s[p++];
        if (p < s.size()) p++;
        return v;
    }
    if (s[p] == 'n') {
        p += 4;
        return {};
    }
    // number
    JVal v;
    v.t = JVal::NUM;
    size_t start = p;
    if (s[p] == '-') p++;
    while (p < s.size() && isdigit(s[p])) p++;
    if (p < s.size() && s[p] == '.') {
        p++;
        while (p < s.size() && isdigit(s[p])) p++;
    }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
        p++;
        if (p < s.size() && (s[p] == '+' || s[p] == '-')) p++;
        while (p < s.size() && isdigit(s[p])) p++;
    }
    v.n = stod(s.substr(start, p - start));
    return v;
}

JVal jparse(const string& s) {
    size_t p = 0;
    return jparse(s, p);
}

// ============================================================
// Problem data structures
// ============================================================

struct Tensor {
    int64_t w, h;
};

struct Op {
    string type;  // "MatMul" or "Pointwise"
    vector<int> ins, outs;
    int64_t base_cost;
};

struct Problem {
    vector<Tensor> tensors;
    vector<Op> ops;
    int64_t fast_cap, slow_bw, nat_w, nat_h;
    // derived
    vector<int> producer;           // producer[t] = op producing tensor t, -1 if graph input
    vector<vector<int>> consumers;  // consumers[t] = ops consuming tensor t
    set<int> graph_ins, graph_outs;
};

Problem read_problem(const char* path) {
    ifstream f(path);
    if (!f) { cerr << "Cannot open " << path << endl; exit(1); }
    stringstream ss;
    ss << f.rdbuf();
    auto j = jparse(ss.str());

    Problem p;
    int nt = j["widths"].sz();
    p.tensors.resize(nt);
    for (int i = 0; i < nt; i++)
        p.tensors[i] = {j["widths"][(size_t)i].i64(), j["heights"][(size_t)i].i64()};

    int no = j["inputs"].sz();
    p.ops.resize(no);
    for (int i = 0; i < no; i++) {
        p.ops[i].type = j["op_types"][(size_t)i].s;
        for (int k = 0; k < j["inputs"][(size_t)i].sz(); k++)
            p.ops[i].ins.push_back((int)j["inputs"][(size_t)i][(size_t)k].i64());
        for (int k = 0; k < j["outputs"][(size_t)i].sz(); k++)
            p.ops[i].outs.push_back((int)j["outputs"][(size_t)i][(size_t)k].i64());
        p.ops[i].base_cost = j["base_costs"][(size_t)i].i64();
    }
    p.fast_cap = j["fast_memory_capacity"].i64();
    p.slow_bw = j["slow_memory_bandwidth"].i64();
    p.nat_w = j["native_granularity"][(size_t)0].i64();
    p.nat_h = j["native_granularity"][(size_t)1].i64();

    // derived
    p.producer.assign(nt, -1);
    p.consumers.resize(nt);
    for (int i = 0; i < no; i++) {
        for (int t : p.ops[i].outs) p.producer[t] = i;
        for (int t : p.ops[i].ins) p.consumers[t].push_back(i);
    }
    for (int i = 0; i < nt; i++) {
        if (p.producer[i] < 0) p.graph_ins.insert(i);
        if (p.consumers[i].empty()) p.graph_outs.insert(i);
    }
    return p;
}

// ============================================================
// Granularity & subgraph analysis
// ============================================================

struct Gran {
    int64_t w, h, k;
};

// Reduction dimension K for a MatMul op (= LHS width = RHS height)
int64_t get_K(const Problem& p, int oi) {
    return p.tensors[p.ops[oi].ins[0]].w;
}

// Instantaneous slice size of a boundary INPUT tensor (for working-set check)
// Takes the max across all consuming ops in the subgraph.
int64_t input_slice(const Problem& p, int tidx, const vector<int>& ops, const Gran& g) {
    int64_t best = 0;
    for (int oi : ops) {
        const Op& op = p.ops[oi];
        for (int j = 0; j < (int)op.ins.size(); j++) {
            if (op.ins[j] == tidx) {
                int64_t s = (op.type == "MatMul")
                    ? (j == 0 ? g.h * g.k : g.w * g.k)
                    : g.w * g.h;
                best = max(best, s);
            }
        }
    }
    return best > 0 ? best : g.w * g.h;
}

// Total memory transferred for a boundary INPUT tensor per spatial tile
// (uses K_full for MatMul; takes max across consuming ops)
int64_t tile_mem_in(const Problem& p, int tidx, const vector<int>& ops, const Gran& g) {
    int64_t best = 0;
    for (int oi : ops) {
        const Op& op = p.ops[oi];
        for (int j = 0; j < (int)op.ins.size(); j++) {
            if (op.ins[j] == tidx) {
                int64_t s;
                if (op.type == "MatMul") {
                    int64_t K = get_K(p, oi);
                    s = j == 0 ? g.h * K : g.w * K;
                } else {
                    s = g.w * g.h;
                }
                best = max(best, s);
            }
        }
    }
    return best > 0 ? best : g.w * g.h;
}

struct SGInfo {
    set<int> in_bd;   // input boundary tensors (need to load)
    set<int> out_bd;  // output boundary tensors (need to evict)
    set<int> ephem;   // ephemeral (internal) tensors
    int64_t out_W, out_H;  // max output tensor dims (for spatial tiling)
};

SGInfo analyze(const Problem& p, const vector<int>& ops) {
    set<int> opset(ops.begin(), ops.end());
    set<int> produced, consumed;
    for (int oi : ops) {
        for (int t : p.ops[oi].outs) produced.insert(t);
        for (int t : p.ops[oi].ins) consumed.insert(t);
    }
    SGInfo info{};
    for (int t : consumed)
        if (!produced.count(t)) info.in_bd.insert(t);
    for (int t : produced) {
        bool external = p.graph_outs.count(t) > 0;
        if (!external)
            for (int c : p.consumers[t])
                if (!opset.count(c)) { external = true; break; }
        if (external)
            info.out_bd.insert(t);
        else if (consumed.count(t))
            info.ephem.insert(t);
        else
            info.out_bd.insert(t);  // dead value, treat as output
    }
    info.out_W = info.out_H = 0;
    for (int oi : ops)
        for (int t : p.ops[oi].outs) {
            info.out_W = max(info.out_W, p.tensors[t].w);
            info.out_H = max(info.out_H, p.tensors[t].h);
        }
    return info;
}

// Working set per tile (must fit in fast_cap)
int64_t working_set(const Problem& p, const vector<int>& ops,
                    const SGInfo& info, const Gran& g) {
    int64_t ws = 0;
    for (int t : info.in_bd) ws += input_slice(p, t, ops, g);
    for ([[maybe_unused]] int t : info.out_bd) ws += g.w * g.h;
    return ws;
}

// ============================================================
// Latency model (per-tile roofline, raster order, no retention)
// ============================================================

double calc_latency(const Problem& p, const vector<int>& ops,
                    const SGInfo& info, const Gran& g) {
    if (info.out_W <= 0 || info.out_H <= 0) return 0;

    int64_t tiles_x = (info.out_W + g.w - 1) / g.w;
    int64_t tiles_y = (info.out_H + g.h - 1) / g.h;
    int64_t ntiles = tiles_x * tiles_y;

    // --- Per spatial tile cost ---
    // Compute: each op runs once per tile, padded to native
    int64_t nat_scale = max((int64_t)1, (g.w + p.nat_w - 1) / p.nat_w) *
                        max((int64_t)1, (g.h + p.nat_h - 1) / p.nat_h);
    double compute = 0;
    for (int oi : ops) compute += (double)p.ops[oi].base_cost;
    compute *= nat_scale;

    // Memory in: total per-tile transfer (full K for MatMul inputs)
    double mem_in = 0;
    for (int t : info.in_bd)
        mem_in += (double)tile_mem_in(p, t, ops, g) / p.slow_bw;
    // Memory out: boundary output tensor slices / bandwidth
    double mem_out = 0;
    for ([[maybe_unused]] int t : info.out_bd)
        mem_out += (double)(g.w * g.h) / p.slow_bw;

    double tile_lat = max(compute, mem_in + mem_out);
    return ntiles * tile_lat;
}

// ============================================================
// Granularity search — find best [w,h,k] for a subgraph
// ============================================================

// Generate power-of-2 candidates up to max_val
vector<int64_t> pow2_candidates(int64_t max_val) {
    vector<int64_t> v;
    for (int64_t x = 1; x <= max_val; x *= 2) v.push_back(x);
    return v;
}

// Returns {best_gran, best_latency}. If nothing fits, returns {{0,0,0}, inf}.
pair<Gran, double> find_best_gran(const Problem& p, const vector<int>& ops) {
    SGInfo info = analyze(p, ops);
    if (info.out_W <= 0) return {{1, 1, 1}, 0};

    // Determine max K across MatMuls (0 if no MatMuls)
    int64_t maxK = 0;
    for (int oi : ops)
        if (p.ops[oi].type == "MatMul")
            maxK = max(maxK, get_K(p, oi));

    auto ws = pow2_candidates(max(info.out_W, info.out_H));
    auto ks = pow2_candidates(max(maxK, (int64_t)1));

    Gran best{0, 0, 0};
    double best_lat = 1e30;

    // Search large→small so we prefer bigger tiles when latency ties
    for (int ki = (int)ks.size() - 1; ki >= 0; ki--) {
        int64_t kv = ks[ki];
        if (maxK > 0 && kv > maxK) continue;
        for (int wi = (int)ws.size() - 1; wi >= 0; wi--) {
            int64_t wv = ws[wi];
            if (wv > info.out_W * 2) continue;
            for (int hi = (int)ws.size() - 1; hi >= 0; hi--) {
                int64_t hv = ws[hi];
                if (hv > info.out_H * 2) continue;
                Gran g{wv, hv, maxK > 0 ? kv : 1};
                if (working_set(p, ops, info, g) > p.fast_cap) continue;
                double lat = calc_latency(p, ops, info, g);
                if (lat < best_lat) {
                    best_lat = lat;
                    best = g;
                }
            }
        }
    }
    return {best, best_lat};
}

// ============================================================
// DAG utilities
// ============================================================

// Topological sort of ops
vector<int> topo_sort(const Problem& p) {
    int n = (int)p.ops.size();
    vector<int> indeg(n, 0);
    vector<vector<int>> adj(n);  // op -> successor ops

    for (int i = 0; i < n; i++)
        for (int t : p.ops[i].outs)
            for (int c : p.consumers[t])
                if (c != i) {
                    adj[i].push_back(c);
                    indeg[c]++;
                }

    // Deduplicate edges (could have multiple tensor connections)
    for (int i = 0; i < n; i++) {
        sort(adj[i].begin(), adj[i].end());
        adj[i].erase(unique(adj[i].begin(), adj[i].end()), adj[i].end());
    }
    // Recount indeg after dedup
    fill(indeg.begin(), indeg.end(), 0);
    for (int i = 0; i < n; i++)
        for (int j : adj[i]) indeg[j]++;

    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indeg[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return order;
}

// ============================================================
// Greedy fusion
// ============================================================

struct Subgraph {
    vector<int> ops;
    Gran gran;
    double latency;
    bool active = true;
    vector<int> retain;      // tensors to keep in fast mem for next SG
    vector<int> traversal;   // tile traversal order (empty = raster/null)
};

// Check if merging sg_a into sg_b would create a cycle in the subgraph DAG
// Returns true if there's a path from a to b NOT using the direct edge
bool creates_cycle(int sg_a, int sg_b,
                   const vector<Subgraph>& sgs,
                   const vector<int>& op_to_sg,
                   const Problem& p) {
    // BFS from sg_a's successors (excluding sg_b) to see if sg_b is reachable
    set<int> visited;
    queue<int> q;

    // Find all successor subgraphs of sg_a
    for (int oi : sgs[sg_a].ops)
        for (int t : p.ops[oi].outs)
            for (int c : p.consumers[t]) {
                int s = op_to_sg[c];
                if (s != sg_a && s != sg_b && sgs[s].active && !visited.count(s)) {
                    visited.insert(s);
                    q.push(s);
                }
            }

    while (!q.empty()) {
        int cur = q.front();
        q.pop();
        // Check if cur reaches sg_b
        for (int oi : sgs[cur].ops)
            for (int t : p.ops[oi].outs)
                for (int c : p.consumers[t]) {
                    int s = op_to_sg[c];
                    if (s == sg_b) return true;
                    if (s != cur && sgs[s].active && !visited.count(s)) {
                        visited.insert(s);
                        q.push(s);
                    }
                }
    }
    return false;
}

// Find adjacent subgraph pairs (producer -> consumer via tensor)
vector<pair<int, int>> adjacent_pairs(const vector<Subgraph>& sgs,
                                      const vector<int>& op_to_sg,
                                      const Problem& p) {
    set<pair<int, int>> pairs;
    for (int si = 0; si < (int)sgs.size(); si++) {
        if (!sgs[si].active) continue;
        for (int oi : sgs[si].ops)
            for (int t : p.ops[oi].outs)
                for (int c : p.consumers[t]) {
                    int sj = op_to_sg[c];
                    if (sj != si && sgs[sj].active)
                        pairs.insert({si, sj});
                }
    }
    return vector<pair<int, int>>(pairs.begin(), pairs.end());
}

vector<Subgraph> greedy_fusion(const Problem& p) {
    int n = (int)p.ops.size();

    // Initialize: each op is its own subgraph
    vector<Subgraph> sgs(n);
    vector<int> op_to_sg(n);
    for (int i = 0; i < n; i++) {
        sgs[i].ops = {i};
        op_to_sg[i] = i;
        auto [g, lat] = find_best_gran(p, sgs[i].ops);
        sgs[i].gran = g;
        sgs[i].latency = lat;
    }

    // Phase 1: merge pairs with positive latency benefit
    bool changed = true;
    while (changed) {
        changed = false;
        auto pairs = adjacent_pairs(sgs, op_to_sg, p);

        int best_a = -1, best_b = -1;
        double best_benefit = 0;
        Gran best_gran{};
        double best_lat = 0;

        for (auto [sa, sb] : pairs) {
            if (creates_cycle(sa, sb, sgs, op_to_sg, p)) continue;

            vector<int> merged_ops = sgs[sa].ops;
            merged_ops.insert(merged_ops.end(), sgs[sb].ops.begin(), sgs[sb].ops.end());

            auto [g, lat] = find_best_gran(p, merged_ops);
            if (g.w == 0) continue;

            double benefit = (sgs[sa].latency + sgs[sb].latency) - lat;
            if (benefit > best_benefit) {
                best_benefit = benefit;
                best_a = sa;
                best_b = sb;
                best_gran = g;
                best_lat = lat;
            }
        }

        if (best_a >= 0 && best_benefit > 0) {
            for (int oi : sgs[best_b].ops) {
                sgs[best_a].ops.push_back(oi);
                op_to_sg[oi] = best_a;
            }
            sgs[best_a].gran = best_gran;
            sgs[best_a].latency = best_lat;
            sgs[best_b].active = false;
            sgs[best_b].ops.clear();
            changed = true;
        }
    }

    // Phase 2: merge pairs with zero latency cost that create ephemeral tensors
    changed = true;
    while (changed) {
        changed = false;
        auto pairs = adjacent_pairs(sgs, op_to_sg, p);

        int best_a = -1, best_b = -1;
        int best_ephem = 0;
        Gran best_gran{};
        double best_lat = 0;

        for (auto [sa, sb] : pairs) {
            if (creates_cycle(sa, sb, sgs, op_to_sg, p)) continue;

            vector<int> merged_ops = sgs[sa].ops;
            merged_ops.insert(merged_ops.end(), sgs[sb].ops.begin(), sgs[sb].ops.end());

            auto [g, lat] = find_best_gran(p, merged_ops);
            if (g.w == 0) continue;

            double benefit = (sgs[sa].latency + sgs[sb].latency) - lat;
            if (benefit < -1e-6) continue;  // don't merge if it increases latency

            int n_ephem = (int)analyze(p, merged_ops).ephem.size();
            if (n_ephem > best_ephem) {
                best_ephem = n_ephem;
                best_a = sa;
                best_b = sb;
                best_gran = g;
                best_lat = lat;
            }
        }

        if (best_a >= 0 && best_ephem > 0) {
            for (int oi : sgs[best_b].ops) {
                sgs[best_a].ops.push_back(oi);
                op_to_sg[oi] = best_a;
            }
            sgs[best_a].gran = best_gran;
            sgs[best_a].latency = best_lat;
            sgs[best_b].active = false;
            sgs[best_b].ops.clear();
            changed = true;
        }
    }

    // Collect active subgraphs
    vector<Subgraph> result;
    for (auto& sg : sgs)
        if (sg.active && !sg.ops.empty()) result.push_back(sg);
    return result;
}

// ============================================================
// Topological sort of subgraphs for output ordering
// ============================================================

vector<int> topo_sort_subgraphs(const vector<Subgraph>& sgs, const Problem& p) {
    int ns = (int)sgs.size();
    // Map each op to its subgraph index in the result vector
    int nops = (int)p.ops.size();
    vector<int> op_to_sg(nops, -1);
    for (int si = 0; si < ns; si++)
        for (int oi : sgs[si].ops)
            op_to_sg[oi] = si;

    // Build subgraph adjacency
    vector<set<int>> adj(ns);
    vector<int> indeg(ns, 0);
    for (int si = 0; si < ns; si++)
        for (int oi : sgs[si].ops)
            for (int t : p.ops[oi].outs)
                for (int c : p.consumers[t]) {
                    int sj = op_to_sg[c];
                    if (sj != si && sj >= 0) adj[si].insert(sj);
                }
    for (int si = 0; si < ns; si++)
        for (int sj : adj[si]) indeg[sj]++;

    queue<int> q;
    for (int i = 0; i < ns; i++)
        if (indeg[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        order.push_back(u);
        for (int v : adj[u])
            if (--indeg[v] == 0) q.push(v);
    }
    return order;
}

// ============================================================
// Zig-zag traversal & retention
// ============================================================

vector<int> gen_zigzag(int64_t tiles_x, int64_t tiles_y) {
    vector<int> order;
    for (int64_t ty = 0; ty < tiles_y; ty++) {
        if (ty % 2 == 0)
            for (int64_t tx = 0; tx < tiles_x; tx++)
                order.push_back((int)(ty * tiles_x + tx));
        else
            for (int64_t tx = tiles_x - 1; tx >= 0; tx--)
                order.push_back((int)(ty * tiles_x + tx));
    }
    return order;
}

bool has_matmul(const Problem& p, const vector<int>& ops) {
    for (int oi : ops)
        if (p.ops[oi].type == "MatMul") return true;
    return false;
}

// 0=other, 1=LHS only, 2=RHS only, 3=both
int matmul_role(const Problem& p, int tidx, const vector<int>& ops) {
    int role = 0;
    for (int oi : ops) {
        const Op& op = p.ops[oi];
        if (op.type == "MatMul") {
            if (!op.ins.empty() && op.ins[0] == tidx) role |= 1;
            if (op.ins.size() > 1 && op.ins[1] == tidx) role |= 2;
        }
    }
    return role;
}

// Latency with zig-zag reuse and retention
double calc_latency_final(const Problem& p, const vector<int>& ops,
                          const SGInfo& info, const Gran& g,
                          bool zigzag,
                          const set<int>& retained_in,
                          const set<int>& retained_out) {
    if (info.out_W <= 0 || info.out_H <= 0) return 0;

    int64_t tiles_x = (info.out_W + g.w - 1) / g.w;
    int64_t tiles_y = (info.out_H + g.h - 1) / g.h;
    int64_t nat_scale = max(1LL, (g.w + p.nat_w - 1) / p.nat_w) *
                        max(1LL, (g.h + p.nat_h - 1) / p.nat_h);

    double compute = 0;
    for (int oi : ops) compute += (double)p.ops[oi].base_cost;
    compute *= nat_scale;

    // mem_out: exclude retained outputs
    double mem_out = 0;
    for (int t : info.out_bd)
        if (!retained_out.count(t))
            mem_out += (double)(g.w * g.h) / p.slow_bw;

    // Classify boundary inputs
    struct TIC { double mem; int role; };  // role: 1=LHS, 2=RHS, 0=other
    vector<TIC> inputs;
    for (int t : info.in_bd) {
        if (retained_in.count(t)) continue;  // free
        int r = matmul_role(p, t, ops);
        double mem = (double)tile_mem_in(p, t, ops, g) / p.slow_bw;
        int rc = (r == 1) ? 1 : (r == 2) ? 2 : 0;
        inputs.push_back({mem, rc});
    }

    // Single tile or no zig-zag: simple model
    if (!zigzag || (tiles_x <= 1 && tiles_y <= 1)) {
        double mem_in = 0;
        for (auto& tic : inputs) mem_in += tic.mem;
        return tiles_x * tiles_y * max(compute, mem_in + mem_out);
    }

    // Per-tile with zig-zag reuse
    double total = 0;
    int64_t prev_tx = -1, prev_ty = -1;
    for (int64_t ty = 0; ty < tiles_y; ty++) {
        bool ltr = (ty % 2 == 0);
        for (int64_t i = 0; i < tiles_x; i++) {
            int64_t tx = ltr ? i : (tiles_x - 1 - i);
            double mem_in = 0;
            for (auto& tic : inputs) {
                bool reuse = false;
                if (prev_tx >= 0) {
                    if (tic.role == 1 && ty == prev_ty) reuse = true;  // LHS, same row
                    if (tic.role == 2 && tx == prev_tx) reuse = true;  // RHS, same col
                }
                if (!reuse) mem_in += tic.mem;
            }
            total += max(compute, mem_in + mem_out);
            prev_tx = tx;
            prev_ty = ty;
        }
    }
    return total;
}

void assign_traversals(vector<Subgraph>& sgs, const Problem& p) {
    for (auto& sg : sgs) {
        if (!has_matmul(p, sg.ops)) continue;
        SGInfo info = analyze(p, sg.ops);
        int64_t tiles_x = (info.out_W + sg.gran.w - 1) / sg.gran.w;
        int64_t tiles_y = (info.out_H + sg.gran.h - 1) / sg.gran.h;
        if (tiles_x * tiles_y > 1)
            sg.traversal = gen_zigzag(tiles_x, tiles_y);
    }
}

void assign_retention(vector<Subgraph>& sgs, const vector<int>& order, const Problem& p) {
    int ns = (int)order.size();
    for (int idx = 0; idx + 1 < ns; idx++) {
        auto& sg_cur = sgs[order[idx]];
        auto& sg_next = sgs[order[idx + 1]];
        SGInfo info_cur = analyze(p, sg_cur.ops);
        SGInfo info_next = analyze(p, sg_next.ops);

        // Candidates: tensors produced by cur and consumed by next
        vector<pair<int, double>> cands;
        for (int t : info_cur.out_bd) {
            if (!info_next.in_bd.count(t)) continue;
            int64_t T_full = p.tensors[t].w * p.tensors[t].h;
            int64_t extra_prod = T_full - sg_cur.gran.w * sg_cur.gran.h;
            int64_t extra_cons = T_full - input_slice(p, t, sg_next.ops, sg_next.gran);
            // Quick feasibility
            int64_t ws_cur = working_set(p, sg_cur.ops, info_cur, sg_cur.gran);
            int64_t ws_next = working_set(p, sg_next.ops, info_next, sg_next.gran);
            if (ws_cur + extra_prod > p.fast_cap) continue;
            if (ws_next + extra_cons > p.fast_cap) continue;
            double benefit = (double)T_full / p.slow_bw * 2.0;
            cands.push_back({t, benefit});
        }

        sort(cands.begin(), cands.end(), [](auto& a, auto& b) {
            return a.second > b.second;
        });

        int64_t used_prod = 0, used_cons = 0;
        int64_t avail_prod = p.fast_cap - working_set(p, sg_cur.ops, info_cur, sg_cur.gran);
        int64_t avail_cons = p.fast_cap - working_set(p, sg_next.ops, info_next, sg_next.gran);
        for (auto& [t, ben] : cands) {
            int64_t T_full = p.tensors[t].w * p.tensors[t].h;
            int64_t ep = T_full - sg_cur.gran.w * sg_cur.gran.h;
            int64_t ec = T_full - input_slice(p, t, sg_next.ops, sg_next.gran);
            if (used_prod + ep <= avail_prod && used_cons + ec <= avail_cons) {
                sg_cur.retain.push_back(t);
                used_prod += ep;
                used_cons += ec;
            }
        }
    }
}

// ============================================================
// Solution output
// ============================================================

void write_solution(const char* path, const vector<Subgraph>& sgs,
                    const vector<int>& order, const Problem& p) {
    ofstream f(path);
    if (!f) { cerr << "Cannot write " << path << endl; exit(1); }

    // Build retained-in sets for each subgraph in schedule order
    int ns = (int)order.size();
    vector<set<int>> retained_in(ns);
    for (int i = 0; i + 1 < ns; i++)
        for (int t : sgs[order[i]].retain)
            retained_in[i + 1].insert(t);

    f << "{\n";

    // subgraphs
    f << "  \"subgraphs\": [";
    for (int i = 0; i < ns; i++) {
        const auto& sg = sgs[order[i]];
        f << (i ? ", " : "") << "[";
        vector<int> sorted_ops = sg.ops;
        sort(sorted_ops.begin(), sorted_ops.end());
        for (int j = 0; j < (int)sorted_ops.size(); j++)
            f << (j ? ", " : "") << sorted_ops[j];
        f << "]";
    }
    f << "],\n";

    // granularities
    f << "  \"granularities\": [";
    for (int i = 0; i < ns; i++) {
        const auto& sg = sgs[order[i]];
        f << (i ? ", " : "") << "[" << sg.gran.w << ", " << sg.gran.h << ", " << sg.gran.k << "]";
    }
    f << "],\n";

    // tensors_to_retain
    f << "  \"tensors_to_retain\": [";
    for (int i = 0; i < ns; i++) {
        const auto& sg = sgs[order[i]];
        f << (i ? ", " : "") << "[";
        for (int j = 0; j < (int)sg.retain.size(); j++)
            f << (j ? ", " : "") << sg.retain[j];
        f << "]";
    }
    f << "],\n";

    // traversal_orders
    f << "  \"traversal_orders\": [";
    for (int i = 0; i < ns; i++) {
        const auto& sg = sgs[order[i]];
        if (sg.traversal.empty()) {
            f << (i ? ", " : "") << "null";
        } else {
            f << (i ? ", " : "") << "[";
            for (int j = 0; j < (int)sg.traversal.size(); j++)
                f << (j ? ", " : "") << sg.traversal[j];
            f << "]";
        }
    }
    f << "],\n";

    // subgraph_latencies — computed with zig-zag + retention
    f << "  \"subgraph_latencies\": [";
    for (int i = 0; i < ns; i++) {
        const auto& sg = sgs[order[i]];
        SGInfo info = analyze(p, sg.ops);
        set<int> r_out(sg.retain.begin(), sg.retain.end());
        bool zz = !sg.traversal.empty();
        double lat = calc_latency_final(p, sg.ops, info, sg.gran,
                                        zz, retained_in[i], r_out);
        f << (i ? ", " : "") << lat;
    }
    f << "]\n";

    f << "}\n";
}

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: ./mlsys <input.json> <output.json>" << endl;
        return 1;
    }

    Problem p = read_problem(argv[1]);

    cerr << "Problem: " << p.tensors.size() << " tensors, "
         << p.ops.size() << " ops, fast_cap=" << p.fast_cap
         << " slow_bw=" << p.slow_bw << " native=[" << p.nat_w << "," << p.nat_h << "]" << endl;

    // Run greedy fusion
    vector<Subgraph> sgs = greedy_fusion(p);
    cerr << "Fusion: " << sgs.size() << " subgraphs" << endl;

    // Topological ordering
    auto order = topo_sort_subgraphs(sgs, p);

    // Assign zig-zag traversal for MatMul subgraphs
    assign_traversals(sgs, p);

    // Assign retention between consecutive subgraphs
    assign_retention(sgs, order, p);

    // Compute final latencies and print summary
    int ns = (int)order.size();
    vector<set<int>> retained_in(ns);
    for (int i = 0; i + 1 < ns; i++)
        for (int t : sgs[order[i]].retain)
            retained_in[i + 1].insert(t);

    double total = 0;
    for (int i = 0; i < ns; i++) {
        auto& sg = sgs[order[i]];
        SGInfo info = analyze(p, sg.ops);
        set<int> r_out(sg.retain.begin(), sg.retain.end());
        bool zz = !sg.traversal.empty();
        double lat = calc_latency_final(p, sg.ops, info, sg.gran,
                                        zz, retained_in[i], r_out);
        total += lat;
        cerr << "  SG[" << i << "] ops=" << sg.ops.size()
             << " gran=[" << sg.gran.w << "," << sg.gran.h << "," << sg.gran.k << "]"
             << (zz ? " zigzag" : "")
             << " retain=" << sg.retain.size()
             << " lat=" << lat << endl;
    }
    cerr << "Total latency: " << total << endl;

    write_solution(argv[2], sgs, order, p);
    cerr << "Solution written to " << argv[2] << endl;
    return 0;
}
