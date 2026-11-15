#pragma once
#include "common.hpp"

struct SimEdge { int u, v, sim; };

struct Graph {
  int n = 0, Delta = 0, K = 0;
  int64_t m = 0;
  vector<Edge> E;
  vector<int64_t> off;
  vector<int> nbr, deg;
  hvec<int> nsr, er;
  vector<float> vals;
  hvec<int> lamv;
  hvec<int> twin;
  vector<int> baseLeafOrder;
  vector<int> erIn;
  vector<uint64_t> RP2, RQ2;
  int thr_rank(const Threshold& eps) const {
    int lo = 0, hi = K;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (cmp_cos2(RP2[mid], RQ2[mid], eps) >= 0) hi = mid;
      else lo = mid + 1;
    }
    return lo;
  }
  float rank_proxy(int rt) const { return vals[rt]; }
  static uint32_t fkey(float s) {
    uint32_t b;
    memcpy(&b, &s, 4);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  }
  void take_exact_ranks() {
    const int64_t mm = E.size();
    er.resize(mm);
    for (int64_t i = 0; i < mm; i++) er[i] = erIn[i];
    vector<int>().swap(erIn);
    vals.resize(K);
    for (int k = 0; k < K; k++) {
      float f = float(sqrt(double(RP2[k]) / double(RQ2[k])));
      if (k > 0 && !(f > vals[k - 1])) f = nextafterf(vals[k - 1], INFINITY);
      vals[k] = f;
    }
  }
  void release_border_arrays() {
    hvec<int>().swap(twin);
    hvec<int>().swap(lamv);
  }
  void finalize(bool fast) {
    m = E.size();
    if (m > INT_MAX / 2) die("the graph has too many edges");
    deg.assign(n, 0);
    for (auto& e : E) {
      deg[e.u]++;
      deg[e.v]++;
    }
    off.assign(n + 1, 0);
    for (int v = 0; v < n; v++) off[v + 1] = off[v] + deg[v];

    take_exact_ranks();
    hvec<SimEdge> se(m);
    {
      vector<int64_t> h(K, 0);
      for (int64_t i = 0; i < m; i++) h[er[i]]++;
      int64_t s = 0;
      for (int r = K - 1; r >= 0; r--) {
        int64_t c = h[r];
        h[r] = s;
        s += c;
      }
      for (int64_t i = 0; i < m; i++) {
        int64_t q = h[er[i]]++;
        se[q] = {E[i].u, E[i].v, er[i]};
      }
    }
    vector<Edge>().swap(E);
    hvec<int>().swap(er);

    {
      const SimEdge* sp = se.data();
      baseLeafOrder = krt_leaf_order(n, m, [sp](int64_t q) { return sp[q].u; },
                                     [sp](int64_t q) { return sp[q].v; });
    }

    if (fast) {
      struct AE { int nbr, nsr, twin, pad; };
      static_assert(sizeof(AE) == 16);
      HugeBuf<AE> ae(off[n]);
      {
        struct VP { int64_t off; int used, pad; };
        hvec<VP> vp(n);
        for (int x = 0; x < n; x++) vp[x] = {off[x], 0, 0};
        constexpr int64_t PD = 16;
        for (int64_t q = 0; q < m; q++) {
          if (q + PD < m) {
            _mm_prefetch((const char*)&vp[se[q + PD].u], _MM_HINT_T0);
            _mm_prefetch((const char*)&vp[se[q + PD].v], _MM_HINT_T0);
          }
          if (q + PD / 2 < m) {
            const SimEdge& f = se[q + PD / 2];
            _mm_prefetch((const char*)&ae[vp[f.u].off + vp[f.u].used], _MM_HINT_T0);
            _mm_prefetch((const char*)&ae[vp[f.v].off + vp[f.v].used], _MM_HINT_T0);
          }
          const SimEdge& e = se[q];
          VP& vpU = vp[e.u];
          VP& vpV = vp[e.v];
          const int ru = vpU.used++, rv = vpV.used++;
          ae[vpU.off + ru] = {e.v, e.sim, rv, 0};
          ae[vpV.off + rv] = {e.u, e.sim, ru, 0};
        }
      }
      hvec<SimEdge>().swap(se);
      nbr.resize(off[n]);
      nsr.resize(off[n]);
      twin.resize(off[n]);
      for (int64_t i = 0; i < off[n]; i++) {
        const AE& x = ae[i];
        nbr[i] = x.nbr;
        nsr[i] = x.nsr;
        twin[i] = x.twin;
      }
    } else {
      nbr.resize(off[n]);
      nsr.resize(off[n]);
      twin.resize(off[n]);
      {
        vector<int> used(n, 0);
        for (int64_t q = 0; q < m; q++) {
          const SimEdge& e = se[q];
          const int ru = used[e.u]++, rv = used[e.v]++;
          const int64_t pu = off[e.u] + ru, pv = off[e.v] + rv;
          nbr[pu] = e.v;
          nsr[pu] = e.sim;
          twin[pu] = rv;
          nbr[pv] = e.u;
          nsr[pv] = e.sim;
          twin[pv] = ru;
        }
      }
      hvec<SimEdge>().swap(se);
    }

    lamv.resize(off[n]);
    for (int v = 0; v < n; v++) {
      const int64_t b = off[v], e = off[v + 1];
      int prevR = -1, prevL = 0;
      for (int64_t i = e - 1; i >= b; i--) {
        prevL = (prevR == nsr[i]) ? prevL : int(i - b) + 2;
        lamv[i] = prevL;
        prevR = nsr[i];
      }
    }
    Delta = n ? *max_element(deg.begin(), deg.end()) : 0;
  }
};

static constexpr size_t kSortSmall = 2048;
static constexpr int kDigitBits = 11;

template <class V, class KeyF>
static void sort_desc_by_rank(V& a, V& tmp, int K, KeyF key, vector<int64_t>& cnt) {
  using T = typename V::value_type;
  const size_t len = a.size();
  if (len <= 32) {
    for (size_t i = 1; i < len; i++) {
      const T x = a[i];
      const auto kx = key(x);
      size_t j = i;
      while (j > 0 && key(a[j - 1]) < kx) {
        a[j] = a[j - 1];
        j--;
      }
      a[j] = x;
    }
    return;
  }
  if (len < kSortSmall) {
    stable_sort(a.begin(), a.end(), [&](const T& x, const T& y) { return key(x) > key(y); });
    return;
  }
  tmp.resize(len);
  auto pass = [&](int shift, int buckets, V& src, V& dst) {
    cnt.assign(buckets, 0);
    for (auto& x : src) cnt[buckets - 1 - ((key(x) >> shift) & (buckets - 1))]++;
    int64_t s = 0;
    for (int b = 0; b < buckets; b++) {
      int64_t c = cnt[b];
      cnt[b] = s;
      s += c;
    }
    for (auto& x : src) dst[cnt[buckets - 1 - ((key(x) >> shift) & (buckets - 1))]++] = x;
  };
  int nbits = 1;
  while ((int64_t(1) << nbits) < K) nbits++;
  if (nbits <= kDigitBits || (nbits <= 16 && len >= (size_t(1) << 16))) {
    pass(0, 1 << nbits, a, tmp);
    a.swap(tmp);
  } else {
    V* src = &a;
    V* dst = &tmp;
    for (int shift = 0; shift < nbits; shift += kDigitBits) {
      pass(shift, 1 << kDigitBits, *src, *dst);
      swap(src, dst);
    }
    if (src != &a) a.swap(tmp);
  }
}

static Graph load_graph(const string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (!fp) die("cannot open " + path);
  Graph G;
  char* line = nullptr;
  size_t cap = 0;
  int n = 0;
  while (getline(&line, &cap, fp) != -1) {
    char *p, *q;
    const long long u = strtoll(line, &p, 10), v = strtoll(p, &q, 10);
    if (q == p) continue;
    if (min(u, v) < 0 || max(u, v) >= (1LL << 30) - 1) die("vertex id out of range in " + path);
    if (u == v) die("self-loop in " + path);
    G.E.push_back({int(u), int(v)});
    n = max(n, int(max(u, v)) + 1);
  }
  free(line);
  fclose(fp);
  G.n = n;
  vector<uint64_t> key;
  key.reserve(G.E.size());
  for (const Edge& e : G.E) key.push_back(uint64_t(min(e.u, e.v)) << 32 | max(e.u, e.v));
  sort(key.begin(), key.end());
  if (adjacent_find(key.begin(), key.end()) != key.end()) die("repeated edge in " + path);
  return G;
}

static void count_common(const Graph& G, vector<int>& c) {
  const int n = G.n;
  const int64_t m = G.E.size();
  const vector<Edge>& E = G.E;
  vector<int> deg(n, 0);
  for (int64_t i = 0; i < m; i++) {
    deg[E[i].u]++;
    deg[E[i].v]++;
  }
  auto low_first = [&](int u, int v) { return deg[u] < deg[v] || (deg[u] == deg[v] && u < v); };

  vector<int64_t> ooff(int64_t(n) + 1, 0);
  vector<int> odg(n, 0);
  for (int64_t i = 0; i < m; i++) odg[low_first(E[i].u, E[i].v) ? E[i].u : E[i].v]++;
  for (int x = 0; x < n; x++) ooff[x + 1] = ooff[x] + odg[x];
  vector<int> onb(m), oe(m);
  {
    vector<int64_t> pos(ooff.begin(), ooff.end());
    for (int64_t i = 0; i < m; i++) {
      const bool f = low_first(E[i].u, E[i].v);
      const int low = f ? E[i].u : E[i].v, high = f ? E[i].v : E[i].u;
      const int64_t p = pos[low]++;
      onb[p] = high;
      oe[p] = int(i);
    }
  }

  c.assign(m, 0);
  vector<int> mark(n, -1);
  for (int u = 0; u < n; u++) {
    const int64_t b = ooff[u], e = ooff[u + 1];
    if (e - b < 2) continue;
    for (int64_t i = b; i < e; i++) mark[onb[i]] = oe[i];
    for (int64_t i = b; i < e; i++) {
      const int v = onb[i], euv = oe[i];
      for (int64_t j = ooff[v]; j < ooff[v + 1]; j++) {
        const int euw = mark[onb[j]];
        if (euw >= 0) {
          c[euv]++;
          c[euw]++;
          c[oe[j]]++;
        }
      }
    }
    for (int64_t i = b; i < e; i++) mark[onb[i]] = -1;
  }
  for (int64_t i = 0; i < m; i++) c[i] += 2;
}

static void compute_similarity(Graph& G) {
  const int64_t m = G.E.size();
  vector<int> c;
  count_common(G, c);
  vector<int64_t> deg(G.n, 0);
  for (auto& e : G.E) {
    deg[e.u]++;
    deg[e.v]++;
  }
  vector<uint64_t> c2(m), ab(m);
  for (int64_t i = 0; i < m; i++) {
    const uint64_t ci = uint32_t(c[i]), a = uint64_t(deg[G.E[i].u] + 1),
                   b = uint64_t(deg[G.E[i].v] + 1);
    c2[i] = ci * ci;
    ab[i] = a * b;
  }
  vector<int>().swap(c);

  G.erIn.assign(m, 0);
  vector<int> ord(m);
  iota(ord.begin(), ord.end(), 0);
  auto sim_less = [&](int i, int j) { return u128(c2[i]) * ab[j] < u128(c2[j]) * ab[i]; };
  sort(ord.begin(), ord.end(), sim_less);
  for (int64_t k = 0; k < m; k++) {
    const int i = ord[k];
    if (k == 0 || sim_less(ord[k - 1], i)) {
      const uint64_t g = std::gcd(c2[i], ab[i]);
      G.RP2.push_back(c2[i] / g);
      G.RQ2.push_back(ab[i] / g);
    }
    G.erIn[i] = int(G.RP2.size()) - 1;
  }
  G.K = int(G.RP2.size());
}
