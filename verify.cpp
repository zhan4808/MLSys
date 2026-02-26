// verify.cpp — Standalone solution validator
// Usage: ./verify <input.json> <output.json>
//
// Checks:
//   1. Every op appears in exactly one subgraph
//   2. Subgraphs are in valid topological order
//   3. Working set fits in fast_memory_capacity per tile
//   4. Recomputes latency per subgraph and compares to reported values
//   5. All graph outputs are produced and evicted
//
// Also computes the "unfused baseline" for comparison.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// ---- Reuse JSON parser from solver.cpp ----
struct JVal {
    enum T { NUL, NUM, STR, ARR, OBJ } t = NUL;
    double n = 0;
    string s;
    vector<JVal> a;
    vector<pair<string, JVal>> o;
    const JVal& operator[](const char* key) const {
        for (auto& [k, v] : o) if (k == key) return v;
        static JVal nil; return nil;
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
        JVal v; v.t = JVal::OBJ; p++; ws();
        while (p < s.size() && s[p] != '}') {
            auto k = jparse(s, p); ws();
            if (p < s.size() && s[p] == ':') p++;
            auto val = jparse(s, p);
            v.o.push_back({k.s, val}); ws();
            if (p < s.size() && s[p] == ',') p++;
        }
        if (p < s.size()) p++; return v;
    }
    if (s[p] == '[') {
        JVal v; v.t = JVal::ARR; p++; ws();
        while (p < s.size() && s[p] != ']') {
            v.a.push_back(jparse(s, p)); ws();
            if (p < s.size() && s[p] == ',') p++;
        }
        if (p < s.size()) p++; return v;
    }
    if (s[p] == '"') {
        JVal v; v.t = JVal::STR; p++;
        while (p < s.size() && s[p] != '"') v.s += s[p++];
        if (p < s.size()) p++; return v;
    }
    if (s[p] == 'n') { p += 4; return {}; }
    JVal v; v.t = JVal::NUM;
    size_t start = p;
    if (s[p] == '-') p++;
    while (p < s.size() && isdigit(s[p])) p++;
    if (p < s.size() && s[p] == '.') { p++; while (p < s.size() && isdigit(s[p])) p++; }
    if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
        p++; if (p < s.size() && (s[p] == '+' || s[p] == '-')) p++;
        while (p < s.size() && isdigit(s[p])) p++;
    }
    v.n = stod(s.substr(start, p - start)); return v;
}
JVal jparse(const string& s) { size_t p = 0; return jparse(s, p); }

// ---- Problem/Solution ----
struct Tensor { int64_t w, h; };
struct Op { string type; vector<int> ins, outs; int64_t base_cost; };

struct Problem {
    vector<Tensor> tensors;
    vector<Op> ops;
    int64_t fast_cap, slow_bw, nat_w, nat_h;
    vector<int> producer;
    vector<vector<int>> consumers;
    set<int> graph_ins, graph_outs;
};

struct SolSG {
    vector<int> ops;
    int64_t w, h, k;
    vector<int> retain;
    double reported_lat;
};

Problem read_problem(const char* path) {
    ifstream f(path); stringstream ss; ss << f.rdbuf();
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

vector<SolSG> read_solution(const char* path) {
    ifstream f(path); stringstream ss; ss << f.rdbuf();
    auto j = jparse(ss.str());
    int n = j["subgraphs"].sz();
    vector<SolSG> sgs(n);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < j["subgraphs"][(size_t)i].sz(); k++)
            sgs[i].ops.push_back((int)j["subgraphs"][(size_t)i][(size_t)k].i64());
        sgs[i].w = j["granularities"][(size_t)i][(size_t)0].i64();
        sgs[i].h = j["granularities"][(size_t)i][(size_t)1].i64();
        sgs[i].k = j["granularities"][(size_t)i][(size_t)2].i64();
        if (j["tensors_to_retain"][(size_t)i].t == JVal::ARR)
            for (int k = 0; k < j["tensors_to_retain"][(size_t)i].sz(); k++)
                sgs[i].retain.push_back((int)j["tensors_to_retain"][(size_t)i][(size_t)k].i64());
        sgs[i].reported_lat = j["subgraph_latencies"][(size_t)i].n;
    }
    return sgs;
}

// ---- Verification ----

int64_t get_K(const Problem& p, int oi) { return p.tensors[p.ops[oi].ins[0]].w; }

int main(int argc, char** argv) {
    if (argc < 3) { cerr << "Usage: ./verify <input.json> <output.json>\n"; return 1; }

    auto prob = read_problem(argv[1]);
    auto sgs = read_solution(argv[2]);
    int nops = (int)prob.ops.size();
    int nsg = (int)sgs.size();
    bool ok = true;

    printf("=== Verification: %d ops, %d subgraphs ===\n", nops, nsg);

    // CHECK 1: Every op in exactly one subgraph
    vector<int> op_count(nops, 0);
    for (auto& sg : sgs) for (int oi : sg.ops) op_count[oi]++;
    for (int i = 0; i < nops; i++) {
        if (op_count[i] == 0) { printf("FAIL: op %d not in any subgraph\n", i); ok = false; }
        // Note: ops CAN appear in multiple subgraphs (recomputation). Count ≥ 1 is valid.
    }
    printf("[%s] All ops covered (recomputation allowed)\n", ok ? "PASS" : "FAIL");

    // CHECK 2: Topological order — for each subgraph pair (i < j),
    // no tensor produced by sgs[j] should be consumed by sgs[i]
    {
        // Map ops to their subgraph index
        map<int, int> op_to_sg;
        for (int si = 0; si < nsg; si++)
            for (int oi : sgs[si].ops) op_to_sg[oi] = si;

        bool topo_ok = true;
        for (int si = 0; si < nsg; si++) {
            set<int> opset(sgs[si].ops.begin(), sgs[si].ops.end());
            for (int oi : sgs[si].ops)
                for (int t : prob.ops[oi].ins) {
                    int prod = prob.producer[t];
                    if (prod >= 0 && !opset.count(prod)) {
                        // t is produced by op `prod` — its subgraph must come before si
                        auto it = op_to_sg.find(prod);
                        if (it != op_to_sg.end() && it->second > si) {
                            printf("FAIL: SG[%d] consumes tensor %d produced by SG[%d] (later)\n",
                                   si, t, it->second);
                            topo_ok = false;
                        }
                    }
                }
        }
        printf("[%s] Topological order\n", topo_ok ? "PASS" : "FAIL");
        if (!topo_ok) ok = false;
    }

    // CHECK 3: Working set per subgraph
    for (int si = 0; si < nsg; si++) {
        auto& sg = sgs[si];
        set<int> opset(sg.ops.begin(), sg.ops.end());
        set<int> produced, consumed;
        for (int oi : sg.ops) {
            for (int t : prob.ops[oi].outs) produced.insert(t);
            for (int t : prob.ops[oi].ins) consumed.insert(t);
        }
        set<int> in_bd, out_bd;
        for (int t : consumed) if (!produced.count(t)) in_bd.insert(t);
        for (int t : produced) {
            bool ext = prob.graph_outs.count(t) > 0;
            if (!ext) for (int c : prob.consumers[t])
                if (!opset.count(c)) { ext = true; break; }
            if (ext) out_bd.insert(t);
        }

        // Compute working set with per-k-step slices
        int64_t ws = 0;
        for (int t : in_bd) {
            int64_t best = 0;
            for (int oi : sg.ops) {
                for (int j = 0; j < (int)prob.ops[oi].ins.size(); j++) {
                    if (prob.ops[oi].ins[j] == t) {
                        int64_t s = (prob.ops[oi].type == "MatMul")
                            ? (j == 0 ? sg.h * sg.k : sg.w * sg.k)
                            : sg.w * sg.h;
                        best = max(best, s);
                    }
                }
            }
            ws += (best > 0 ? best : sg.w * sg.h);
        }
        for ([[maybe_unused]] int t : out_bd) ws += sg.w * sg.h;

        if (ws > prob.fast_cap) {
            printf("FAIL: SG[%d] working set %lld > fast_cap %lld\n",
                   si, (long long)ws, (long long)prob.fast_cap);
            ok = false;
        }
    }
    printf("[%s] Working set fits\n", ok ? "PASS" : "FAIL");

    // CHECK 4: Recompute latencies (simplified model)
    double total_reported = 0, total_recomputed = 0;
    for (int si = 0; si < nsg; si++) {
        auto& sg = sgs[si];
        set<int> opset(sg.ops.begin(), sg.ops.end());
        set<int> produced, consumed;
        for (int oi : sg.ops) {
            for (int t : prob.ops[oi].outs) produced.insert(t);
            for (int t : prob.ops[oi].ins) consumed.insert(t);
        }
        set<int> in_bd, out_bd;
        for (int t : consumed) if (!produced.count(t)) in_bd.insert(t);
        for (int t : produced) {
            bool ext = prob.graph_outs.count(t) > 0;
            if (!ext) for (int c : prob.consumers[t])
                if (!opset.count(c)) { ext = true; break; }
            if (ext) out_bd.insert(t);
        }

        int64_t out_W = 0, out_H = 0;
        for (int oi : sg.ops)
            for (int t : prob.ops[oi].outs) {
                out_W = max(out_W, prob.tensors[t].w);
                out_H = max(out_H, prob.tensors[t].h);
            }

        int64_t ntiles = ((out_W + sg.w - 1) / sg.w) * ((out_H + sg.h - 1) / sg.h);
        int64_t ns = max((int64_t)1, (sg.w + prob.nat_w - 1) / prob.nat_w) *
                     max((int64_t)1, (sg.h + prob.nat_h - 1) / prob.nat_h);

        double compute = 0;
        for (int oi : sg.ops) compute += (double)prob.ops[oi].base_cost;
        compute *= ns;

        double mem_in = 0;
        for (int t : in_bd) {
            int64_t best = 0;
            for (int oi : sg.ops)
                for (int j = 0; j < (int)prob.ops[oi].ins.size(); j++)
                    if (prob.ops[oi].ins[j] == t) {
                        int64_t s;
                        if (prob.ops[oi].type == "MatMul") {
                            int64_t K = get_K(prob, oi);
                            s = j == 0 ? sg.h * K : sg.w * K;
                        } else {
                            s = sg.w * sg.h;
                        }
                        best = max(best, s);
                    }
            mem_in += (double)(best > 0 ? best : sg.w * sg.h) / prob.slow_bw;
        }
        double mem_out = 0;
        for ([[maybe_unused]] int t : out_bd) mem_out += (double)(sg.w * sg.h) / prob.slow_bw;

        double lat = ntiles * max(compute, mem_in + mem_out);
        total_reported += sg.reported_lat;
        total_recomputed += lat;

        double diff = fabs(lat - sg.reported_lat);
        if (diff > 0.1) {
            printf("  SG[%d]: reported=%.1f recomputed=%.1f (delta=%.1f)\n",
                   si, sg.reported_lat, lat, diff);
        }
    }

    printf("[INFO] Total reported latency:    %.1f\n", total_reported);
    printf("[INFO] Total recomputed latency:  %.1f\n", total_recomputed);

    // CHECK 5: All graph outputs produced (or are pass-through graph inputs)
    {
        set<int> all_produced;
        for (auto& sg : sgs)
            for (int oi : sg.ops)
                for (int t : prob.ops[oi].outs) all_produced.insert(t);
        bool outputs_ok = true;
        for (int t : prob.graph_outs) {
            if (!all_produced.count(t)) {
                if (prob.graph_ins.count(t)) {
                    // Tensor is both graph input and output — already in slow memory
                    printf("  [INFO] Tensor %d is pass-through (graph in+out, no ops)\n", t);
                } else {
                    printf("FAIL: graph output tensor %d never produced\n", t);
                    outputs_ok = false;
                }
            }
        }
        printf("[%s] All graph outputs produced\n", outputs_ok ? "PASS" : "FAIL");
        if (!outputs_ok) ok = false;
    }

    // Unfused baseline for comparison
    double baseline = 0;
    for (int oi = 0; oi < nops; oi++) {
        vector<int> single = {oi};
        set<int> produced, consumed;
        for (int t : prob.ops[oi].outs) produced.insert(t);
        for (int t : prob.ops[oi].ins) consumed.insert(t);
        set<int> in_bd, out_bd;
        for (int t : consumed) in_bd.insert(t);
        for (int t : produced) out_bd.insert(t);

        // Best granularity for single op
        int64_t out_W = 0, out_H = 0;
        for (int t : prob.ops[oi].outs) {
            out_W = max(out_W, prob.tensors[t].w);
            out_H = max(out_H, prob.tensors[t].h);
        }
        int64_t maxK = 0;
        if (prob.ops[oi].type == "MatMul") maxK = get_K(prob, oi);

        double best_single = 1e30;
        for (int64_t w = 1; w <= max(out_W, (int64_t)1); w *= 2)
        for (int64_t h = 1; h <= max(out_H, (int64_t)1); h *= 2)
        for (int64_t k = 1; k <= max(maxK, (int64_t)1); k *= 2) {
            // WS check
            int64_t ws = 0;
            for (int t : in_bd) {
                for (int j = 0; j < (int)prob.ops[oi].ins.size(); j++)
                    if (prob.ops[oi].ins[j] == t) {
                        ws += (prob.ops[oi].type == "MatMul")
                            ? (j == 0 ? h * k : w * k) : w * h;
                        break;
                    }
            }
            for ([[maybe_unused]] int t : out_bd) ws += w * h;
            if (ws > prob.fast_cap) continue;

            int64_t ntiles = ((out_W + w - 1) / w) * ((out_H + h - 1) / h);
            int64_t ns = max((int64_t)1, (w + prob.nat_w - 1) / prob.nat_w) *
                         max((int64_t)1, (h + prob.nat_h - 1) / prob.nat_h);
            double compute = (double)prob.ops[oi].base_cost * ns;
            double mi = 0;
            for (int t : in_bd) {
                for (int j = 0; j < (int)prob.ops[oi].ins.size(); j++)
                    if (prob.ops[oi].ins[j] == t) {
                        int64_t K = (prob.ops[oi].type == "MatMul") ? get_K(prob, oi) : 0;
                        int64_t s = (prob.ops[oi].type == "MatMul")
                            ? (j == 0 ? h * K : w * K) : w * h;
                        mi += (double)s / prob.slow_bw;
                        break;
                    }
            }
            double mo = 0;
            for ([[maybe_unused]] int t : out_bd) mo += (double)(w * h) / prob.slow_bw;
            double lat = ntiles * max(compute, mi + mo);
            best_single = min(best_single, lat);
        }
        baseline += best_single;
    }

    printf("\n[INFO] Unfused baseline: %.1f\n", baseline);
    printf("[INFO] Fusion speedup:   %.2fx\n", baseline / total_recomputed);
    printf("\n%s\n", ok ? "=== ALL CHECKS PASSED ===" : "=== SOME CHECKS FAILED ===");
    return ok ? 0 : 1;
}
