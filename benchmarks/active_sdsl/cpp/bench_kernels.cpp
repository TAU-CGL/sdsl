// Native kernel benchmark for the active-SDSL hot paths.
//
// Includes the sdsl core headers UNMODIFIED. Variants that would require a core
// change (e.g. pre-reserving the split output) are implemented here, in the
// benchmark, as "mirror" re-implementations of localize_omp_forkjoin, and their
// outputs are checked against the original for equality.
//
// Usage: bench_kernels <inputs_dir> <out.json> [quick]
// Thread count follows OMP_NUM_THREADS (localize_omp_forkjoin uses omp_get_max_threads()).
#ifndef SDSL_CPP_ONLY
#define SDSL_CPP_ONLY
#endif
#include <sdsl/sdsl.hpp>
#include <sdsl/environments/env_pgm.hpp>
#include <sdsl/fusion/fusion_2d.hpp>
#include <sdsl/fusion/metrics.hpp>
#include <sdsl/cleanup.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace sdsl;
using V3 = Voxel<3, double>;
using C3 = Configuration<3, double>;
using Clock = std::chrono::steady_clock;

static double now_s() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

// ----------------------------------------------------------------------------- inputs
struct MapInput {
    std::string name;
    int width = 0, height = 0;
    double res = 0, ox = 0, oy = 0, occ = 0.65;
    int negate = 0;
    std::vector<uint8_t> grid;
    std::vector<C3> poses;
};

static bool readMap(const std::string& path, MapInput& m) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[4];  // width, height, negate, n_poses
    double d[4];     // res, ox, oy, occ
    f.read((char*)hdr, sizeof(hdr));
    f.read((char*)d, sizeof(d));
    m.width = hdr[0]; m.height = hdr[1]; m.negate = hdr[2];
    m.res = d[0]; m.ox = d[1]; m.oy = d[2]; m.occ = d[3];
    for (int i = 0; i < hdr[3]; ++i) {
        double p[3];
        f.read((char*)p, sizeof(p));
        m.poses.emplace_back(p[0], p[1], p[2]);
    }
    m.grid.resize((size_t)m.width * m.height);
    f.read((char*)m.grid.data(), m.grid.size());
    return (bool)f;
}

// ----------------------------------------------------------------------------- json
struct Json {
    std::ostringstream s;
    bool first = true;
    void key(const std::string& k) { if (!first) s << ",\n"; first = false; s << "\"" << k << "\": "; }
    void num(const std::string& k, double v) { key(k); s << v; }
    void raw(const std::string& k, const std::string& v) { key(k); s << v; }
};

static std::string vecJson(const std::vector<double>& v) {
    std::ostringstream s; s << "[";
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    s << "]"; return s.str();
}
static std::string ivecJson(const std::vector<size_t>& v) {
    std::ostringstream s; s << "[";
    for (size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << v[i];
    s << "]"; return s.str();
}

// ----------------------------------------------------------------------------- mirror localizer
struct Phases {
    double filter = 0, split = 0, verify = 0;
    std::vector<size_t> iterCounts;
    bool aborted = false;  // voxel cap exceeded (pose too ambiguous to finish safely)
};

// Safety caps (apply to every host): beyond MAX_VOXELS entering an iteration the run is
// aborted (memory: the RPi has 4 GB and no swap); if any split would produce more than
// MAX_QUADRATIC_SPLIT voxels, the ORIGINAL O(N^2) split is not run (it can take hours).
static const size_t MAX_VOXELS = 2000000;
static const size_t MAX_QUADRATIC_SPLIT = 150000;

// Same algorithm as sdsl::localize_omp_forkjoin (timeout=0), with optional fixes:
//   preReserve      : reserve the split output once per iteration (fixes O(N^2) growth)
//   skipFinalSplit  : do not split after the last filter (output is unused)
//   doVerify        : run the (discarded) verify() pass like the original
template<int D, typename FT, Predicate<D, FT> Pred>
std::vector<Voxel<D, FT>> localizeMirror(Voxel<D, FT> bbox, Pred predicate, int depth, Phases& ph,
                                         bool preReserve, bool skipFinalSplit, bool doVerify) {
    std::vector<Voxel<D, FT>> voxels, localization, cleaned;
    voxels.push_back(bbox);
    for (int i = 0; i < depth; ++i) {
        predicate.updateIteration(i);
        ph.iterCounts.push_back(voxels.size());
        if (voxels.size() > MAX_VOXELS) { ph.aborted = true; return {}; }
        localization.clear();
        double t0 = now_s();
        #pragma omp parallel
        {
            std::vector<Voxel<D, FT>> tmp;
            #pragma omp for nowait
            for (int v_i = 0; v_i < (int)voxels.size(); ++v_i) {
                auto v = voxels[v_i];
                if (predicate(v)) tmp.push_back(v);
            }
            #pragma omp critical
            { localization.insert(localization.end(), tmp.begin(), tmp.end()); }
        }
        double t1 = now_s();
        ph.filter += t1 - t0;
        voxels.clear();
        if (skipFinalSplit && i == depth - 1) break;
        if (preReserve) voxels.reserve(localization.size() << D);
        for (auto v : localization) v.split(voxels);
        ph.split += now_s() - t1;
    }
    if (doVerify) {
        double t0 = now_s();
        for (auto v : localization) if (predicate.verify(v)) cleaned.push_back(v);
        ph.verify += now_s() - t0;
    }
    return localization;
}

static std::vector<std::array<double, 6>> canon(const std::vector<V3>& v) {
    std::vector<std::array<double, 6>> c;
    c.reserve(v.size());
    for (auto& x : v) c.push_back({x.bottomLeft[0], x.bottomLeft[1], x.bottomLeft[2],
                                   x.topRight[0], x.topRight[1], x.topRight[2]});
    std::sort(c.begin(), c.end());
    return c;
}

template<typename F>
static double bestOf(int reps, F&& f) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        double t0 = now_s();
        f();
        best = std::min(best, now_s() - t0);
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s inputs_dir out.json [quick]\n", argv[0]); return 2; }
    std::string dir = argv[1], outPath = argv[2];
    bool quick = argc > 3 && std::string(argv[3]) == "quick";
    const int N_RAYS = 16, DEPTH = 10;
    const double KK = 0.7, ERR = 0.015;
    const char* repsEnv = std::getenv("SDSL_BENCH_REPS");
    const int REPS = quick ? 1 : (repsEnv ? std::max(1, std::atoi(repsEnv)) : 3);
    std::vector<std::string> maps = quick ? std::vector<std::string>{"symmetry_2"}
                                          : std::vector<std::string>{"square", "symmetry_2", "plus", "apt"};

    Json out;
    out.num("omp_max_threads", omp_get_max_threads());
    std::ostringstream locJ; locJ << "[";
    bool firstLoc = true;
    std::vector<V3> pool;  // raw localization outputs for belief-kernel scaling

    for (auto& mname : maps) {
        MapInput m;
        if (!readMap(dir + "/" + mname + ".bin", m)) { std::fprintf(stderr, "cannot read %s\n", mname.c_str()); return 1; }
        auto env = std::make_shared<Env_PGM<3>>();
        env->load(m.grid.data(), m.width, m.height, m.res, m.ox, m.oy, m.occ, m.negate != 0);
        V3 bbox = env->boundingBox();
        int np = quick ? 1 : (int)m.poses.size();
        for (int pi = 0; pi < np; ++pi) {
            C3 q = m.poses[pi];
            std::vector<C3> odo; std::vector<double> meas;
            for (int k = 0; k < N_RAYS; ++k) {
                double dt = 2.0 * M_PI * k / N_RAYS;
                odo.emplace_back(0.0, 0.0, dt);
                meas.push_back(env->measureDistance(C3(q[0], q[1], q[2] + dt)));
            }
            Predicate_Fwd2D<3, double> pred(env, odo, meas, KK, ERR);

            std::vector<V3> resOrig, resMirror, resFixed, resFixedAll;
            // 1) fully fixed variant first: cheap, and tells us how big the splits get
            Phases phA;
            double tFixedAll = bestOf(1, [&] { phA = Phases(); resFixedAll = localizeMirror<3, double>(bbox, pred, DEPTH, phA, true, true, false); });
            if (phA.aborted) {
                std::ostringstream r;
                r << "{\"map\":\"" << mname << "\",\"pose\":[" << q[0] << "," << q[1] << "," << q[2] << "]"
                  << ",\"aborted\":true,\"reason\":\"more than " << MAX_VOXELS << " voxels entering an iteration\""
                  << ",\"iter_counts\":" << ivecJson(phA.iterCounts) << "}";
                locJ << (firstLoc ? "" : ",\n") << r.str();
                firstLoc = false;
                std::fprintf(stderr, "[%s pose %d] ABORTED (voxel cap), iter counts so far: %zu levels\n", mname.c_str(), pi, phA.iterCounts.size());
                continue;
            }
            if (REPS > 1) tFixedAll = std::min(tFixedAll, bestOf(REPS - 1, [&] { phA = Phases(); resFixedAll = localizeMirror<3, double>(bbox, pred, DEPTH, phA, true, true, false); }));
            size_t maxSplit = 0;  // largest split output of the ORIGINAL algorithm (incl. its final, wasted split)
            for (size_t k = 1; k < phA.iterCounts.size(); ++k) maxSplit = std::max(maxSplit, phA.iterCounts[k]);
            maxSplit = std::max(maxSplit, resFixedAll.size() * 8);
            bool runQuadratic = maxSplit <= MAX_QUADRATIC_SPLIT;
            double tOrig = -1, tMirror = -1;
            Phases ph;
            if (runQuadratic) {
                tOrig = bestOf(REPS, [&] { resOrig = localize_omp_forkjoin<3, double>(bbox, pred, DEPTH, 0.0, false); });
                tMirror = bestOf(1, [&] { resMirror = localizeMirror<3, double>(bbox, pred, DEPTH, ph, false, false, true); });
            } else {
                resOrig = resFixedAll;
                resMirror = resFixedAll;
            }
            Phases phF;
            double tFixed = bestOf(REPS, [&] { phF = Phases(); resFixed = localizeMirror<3, double>(bbox, pred, DEPTH, phF, true, false, true); });
            bool same = canon(resOrig) == canon(resMirror) && canon(resOrig) == canon(resFixed)
                        && canon(resOrig) == canon(resFixedAll);

            // belief-layer kernels on this real raw output
            // cleanup is O(n^2) all-pairs: only time it when n_raw is moderate
            const size_t MAX_CLEANUP_N = 40000;
            double tCleanup = -1;
            std::vector<V3> cleaned;
            if (resOrig.size() <= MAX_CLEANUP_N) {
                tCleanup = bestOf(REPS, [&] { auto r = cleanup<3, double>(resOrig, {false, false, true}, 2.0 * M_PI, 1e-9, 3); (void)r; });
                cleaned = cleanup<3, double>(resOrig, {false, false, true}, 2.0 * M_PI, 1e-9, 3);
            }
            std::vector<double> belC(cleaned.size(), cleaned.empty() ? 0.0 : 1.0 / cleaned.size());
            C3 U(0.1, 0.02, 0.05);
            double tFusion = cleaned.empty() ? 0 : bestOf(REPS, [&] { auto r = fusion_2d<double>(cleaned, belC, cleaned, U, 0.015); (void)r; });
            double tCC = cleaned.empty() ? 0 : bestOf(REPS, [&] { auto r = connectedComponentsWithBeliefs<3, double>(cleaned, {false, false, true}, 2.0 * M_PI, 1e-9, true, &belC); (void)r; });
            if (pool.size() < 6400) pool.insert(pool.end(), resOrig.begin(), resOrig.begin() + std::min(resOrig.size(), (size_t)6400));

            std::ostringstream r;
            r << "{\"map\":\"" << mname << "\",\"pose\":[" << q[0] << "," << q[1] << "," << q[2] << "]"
              << ",\"n_raw\":" << resOrig.size() << ",\"n_clean\":" << cleaned.size()
              << ",\"t_orig\":" << tOrig << ",\"t_mirror\":" << tMirror
              << ",\"mirror_filter\":" << ph.filter << ",\"mirror_split\":" << ph.split << ",\"mirror_verify\":" << ph.verify
              << ",\"t_fixed_reserve\":" << tFixed << ",\"fixed_filter\":" << phF.filter << ",\"fixed_split\":" << phF.split
              << ",\"fixed_verify\":" << phF.verify
              << ",\"t_fixed_all\":" << tFixedAll << ",\"fixedall_filter\":" << phA.filter << ",\"fixedall_split\":" << phA.split
              << ",\"iter_counts\":" << ivecJson(phA.iterCounts)
              << ",\"max_split\":" << maxSplit << ",\"quadratic_skipped\":" << (runQuadratic ? "false" : "true")
              << ",\"outputs_identical\":" << (same ? "true" : "false")
              << ",\"t_cleanup_raw\":" << tCleanup << ",\"t_fusion_clean\":" << tFusion << ",\"t_cc_clean\":" << tCC << "}";
            locJ << (firstLoc ? "" : ",\n") << r.str();
            firstLoc = false;
            std::fprintf(stderr, "[%s pose %d] n_raw=%zu orig=%.4fs (filter %.4f split %.4f verify %.4f) fixed=%.4fs fixedAll=%.4fs same=%d cleanup=%.4fs\n",
                         mname.c_str(), pi, resOrig.size(), tOrig, ph.filter, ph.split, ph.verify, tFixed, tFixedAll, same, tCleanup);
        }
        // per-call primitive costs (single thread)
        if (mname == maps.back() || quick) {
            std::vector<C3> qs;
            for (auto& p : m.poses) qs.push_back(p);
            int N = 200000;
            volatile double sink = 0;
            double t = bestOf(3, [&] { for (int i = 0; i < N; ++i) sink += env->measureDistance(qs[i % qs.size()] + 0.0 * i); });
            out.num("measureDistance_ns_" + mname, 1e9 * t / N);
            std::vector<V3> vs;
            for (auto& p : qs) vs.emplace_back(C3(p[0] - 0.05, p[1] - 0.05, 0.0), C3(p[0] + 0.05, p[1] + 0.05, 0.1));
            volatile bool bs = false;
            t = bestOf(3, [&] { for (int i = 0; i < N; ++i) bs = env->intersects(vs[i % vs.size()]); });
            out.num("intersects_ns_" + mname, 1e9 * t / N);
        }
    }
    locJ << "]";
    out.raw("localize", locJ.str());

    // AlwaysTrue split scaling: original vs pre-reserved mirror
    {
        std::ostringstream s; s << "[";
        V3 unit(C3(0, 0, 0), C3(1, 1, 1));
        Predicate_AlwaysTrue<3, double> at;
        int maxOrig = 5, maxFixed = quick ? 5 : 7;
        for (int d = 2; d <= maxFixed; ++d) {
            double tO = -1;
            if (d <= maxOrig) tO = bestOf(REPS, [&] { auto r = localize_omp_forkjoin<3, double>(unit, at, d, 0.0, false); (void)r; });
            Phases p1, p2;
            double tF = bestOf(REPS, [&] { p1 = Phases(); auto r = localizeMirror<3, double>(unit, at, d, p1, true, false, true); (void)r; });
            double tFA = bestOf(REPS, [&] { p2 = Phases(); auto r = localizeMirror<3, double>(unit, at, d, p2, true, true, false); (void)r; });
            s << (d > 2 ? "," : "") << "{\"depth\":" << d << ",\"t_orig\":" << tO << ",\"t_fixed_reserve\":" << tF
              << ",\"t_fixed_all\":" << tFA << "}";
            std::fprintf(stderr, "[alwaystrue d=%d] orig=%.5f fixed=%.5f fixedAll=%.5f\n", d, tO, tF, tFA);
        }
        s << "]";
        out.raw("alwaystrue", s.str());
    }

    // Belief kernels scaling on pooled real voxels (pure C++, no Python conversion)
    {
        std::ostringstream s; s << "{";
        std::vector<size_t> ns = quick ? std::vector<size_t>{100, 200, 400} : std::vector<size_t>{100, 200, 400, 800, 1600, 3200, 6400};
        C3 U(0.1, 0.02, 0.05);
        bool firstK = true;
        auto kernel = [&](const std::string& name, std::function<void(const std::vector<V3>&, const std::vector<double>&)> fn) {
            s << (firstK ? "" : ",") << "\"" << name << "\":[";
            firstK = false;
            bool f2 = true;
            for (size_t n : ns) {
                if (n > pool.size()) break;
                std::vector<V3> v(pool.begin(), pool.begin() + n);
                std::vector<double> b(n, 1.0 / n);
                double t = bestOf(REPS, [&] { fn(v, b); });
                s << (f2 ? "" : ",") << "{\"n\":" << n << ",\"s\":" << t << "}";
                f2 = false;
                if (t > (quick ? 2.0 : 30.0)) break;
            }
            s << "]";
        };
        kernel("cleanup_SE2", [&](auto& v, auto& b) { auto r = cleanup<3, double>(v, {false, false, true}, 2.0 * M_PI, 1e-9, 3); (void)r; });
        kernel("connectedComponents(use3rdDim)", [&](auto& v, auto& b) { auto r = connectedComponentsWithBeliefs<3, double>(v, {false, false, true}, 2.0 * M_PI, 1e-9, true, &b); (void)r; });
        kernel("fusion_2d(n x n)", [&](auto& v, auto& b) { auto r = fusion_2d<double>(v, b, v, U, 0.015); (void)r; });
        kernel("entropy_SE2", [&](auto& v, auto& b) { auto r = entropy_SE2<double>(v, b); (void)r; });
        s << "}";
        out.raw("belief_scaling", s.str());
        out.num("pool_size", (double)pool.size());
    }

    std::ofstream f(outPath);
    f << "{\n" << out.s.str() << "\n}\n";
    std::fprintf(stderr, "wrote %s\n", outPath.c_str());
    return 0;
}
