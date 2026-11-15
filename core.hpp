#pragma once
#include "levels.hpp"

struct Borders;
struct BlockIn;

inline uint8_t sim_key8(float x) {
  const double y = double(x) * 254.0;
  if (y >= 254.0) return 255;
  return uint8_t(1 + int(y));
}

struct PackedU {
  vector<uint8_t> b;
  int w = 4;
  uint64_t mask = 0xffffffffull;
  void init(size_t n, uint64_t maxv) {
    w = (bitsFor(maxv) + 7) / 8;
    mask = w >= 8 ? ~0ull : ((uint64_t(1) << (8 * w)) - 1);
    b.assign(n * w + 8, 0);
  }
  void set(size_t i, uint64_t x) {
    for (int k = 0; k < w; k++) b[i * w + k] = uint8_t(x >> (8 * k));
  }
  uint64_t get(size_t i) const { return load8(b.data() + i * w) & mask; }
};

struct BitArr {
  vector<uint8_t> b;
  uint8_t w = 0;
  uint64_t mask = 0;
  void init(size_t n, int width) {
    w = uint8_t(width);
    mask = (uint64_t(1) << w) - 1;
    b.assign((uint64_t(n) * w + 7) / 8 + 8, 0);
  }
  void set(size_t i, uint64_t x) {
    const uint64_t bit = uint64_t(i) * w;
    uint8_t* q = b.data() + (bit >> 3);
    const uint64_t y = load8(q) | ((x & mask) << (bit & 7));
    memcpy(q, &y, 8);
  }
  uint64_t get(size_t i) const {
    const uint64_t bit = uint64_t(i) * w;
    return (load8(b.data() + (bit >> 3)) >> (bit & 7)) & mask;
  }
};

static bool radix_items(pair<uint32_t, int>* a, size_t k, vector<pair<uint32_t, int>>& tmp) {
  uint32_t mn = UINT32_MAX, mx = 0;
  for (size_t i = 0; i < k; i++) {
    mn = min(mn, a[i].first);
    mx = max(mx, a[i].first);
  }
  const int bits = bitsFor(uint64_t(mx - mn));
  const int passes = (bits + kDigitBits - 1) / kDigitBits;
  const int dw = (bits + passes - 1) / passes;
  if ((size_t(1) << dw) > 2 * k) return false;
  const uint32_t mask = (uint32_t(1) << dw) - 1;
  tmp.resize(k);
  pair<uint32_t, int>* src = a;
  pair<uint32_t, int>* dst = tmp.data();
  uint32_t cnt[1 << kDigitBits];
  for (int p = 0; p < passes; p++) {
    const int sh = p * dw;
    memset(cnt, 0, sizeof(uint32_t) * (size_t(mask) + 1));
    for (size_t i = 0; i < k; i++) cnt[((src[i].first - mn) >> sh) & mask]++;
    uint32_t acc = 0;
    for (uint32_t d = 0; d <= mask; d++) {
      const uint32_t c = cnt[d];
      cnt[d] = acc;
      acc += c;
    }
    for (size_t i = 0; i < k; i++) dst[cnt[((src[i].first - mn) >> sh) & mask]++] = src[i];
    swap(src, dst);
  }
  if (src != a) memcpy(a, src, k * sizeof(pair<uint32_t, int>));
  return true;
}

struct CoreBase {
  virtual ~CoreBase() {}
  static constexpr int lfb = 6;
  static constexpr int kEncRadix = 256;
  int n = 0;
  PackedU orig;
  vector<int> nidv;
  vector<int> oidv;
  vector<uint32_t> amax;
  virtual size_t core_phase(int epsRank, int mu, int* label, vector<pair<int, int>>& out, vector<int>& lcore,
                            int qk) const = 0;
  virtual void init_ids(const Graph& G, const vector<int>& baseLeafOrder) = 0;
  virtual void prepare_lcodes(const Graph& G, const Borders& D) = 0;
  virtual void encode_level(int mu, const Level& l) = 0;
  virtual void release_lcodes() = 0;
  virtual void free_scratch() = 0;
  virtual void finish_level(int mu, BlockOut& o) = 0;
  virtual void save_tail(BlockOut& o) const = 0;
  virtual void load(BlockIn& in, const Graph& G, uint64_t end) = 0;
  virtual int rbytes() const = 0;
  int og(uint32_t v) const { return int(load8(orig.b.data() + size_t(v) * orig.w) & orig.mask); }
};

template <class R>
struct CompactCore : CoreBase {
  int K = 0, Delta = 0;
  struct Lvl {
    vector<R> ss, se;
    PackedU sv;
    vector<R> kw, kpw;
    struct FD { int32_t base; uint32_t off; uint8_t w, pad0, pad1, pad2; };
    uint32_t nlf = 0;
    vector<FD> fd;
    vector<uint8_t> fst;
    BitArr ks, kc;
    BitArr lql, lqs;
    vector<uint8_t> lkl, lks;
  };
  vector<Lvl> levels;
  vector<uint8_t> kmin;

  int rbytes() const override { return int(sizeof(R)); }

  uint32_t leaf(const Lvl& c, uint32_t p) const {
    const uint32_t k = p >> lfb, j = p & ((uint32_t(1) << lfb) - 1);
    const typename Lvl::FD d = c.fd[k];
    const uint64_t bit = uint64_t(j) * d.w;
    const uint64_t x = (load8(c.fst.data() + d.off + (bit >> 3)) >> (bit & 7)) & ((uint64_t(1) << d.w) - 1);
    return uint32_t(d.base + int32_t(j) + int32_t(x));
  }

  void init_ids(const Graph& G, const vector<int>& baseLeafOrder) override {
    n = G.n;
    K = G.K;
    Delta = G.Delta;
    nidv.assign(n, -1);
    oidv = baseLeafOrder;
    orig.init(n, n - 1);
    for (int p = 0; p < n; p++) {
      nidv[baseLeafOrder[p]] = p;
      orig.set(p, baseLeafOrder[p]);
    }
    levels.assign(Delta + 2, Lvl());
    amax.assign(levels.size(), 0);
    kmin.assign(levels.size(), 255);
  }

  struct EncScratch {
    vector<uint32_t> lid, minp;
    vector<int> idx, tmp, cnt, nlo, nhi, par, chOff, chList, pos, order, newlo, nodeOf, stk;
    vector<int64_t> itOff;
    vector<pair<uint32_t, int>> items, itmp;
  };
  inline static EncScratch S;
  void free_scratch() override { S = EncScratch(); }

  void encode_level(int mu, const Level& l) override {
    Lvl& c = levels[mu];
    const size_t ns = l.sv.size();
    if (mu == 1) {
      vector<uint32_t> cnt(size_t(K) + 2, 0);
      vector<R> e(ns);
      for (size_t i = 0; i < ns; i++) {
        e[i] = R(l.se[i] + 1);
        cnt[size_t(e[i]) + 1]++;
      }
      for (size_t r = 0; r + 1 < cnt.size(); r++) cnt[r + 1] += cnt[r];
      c.se.resize(ns);
      c.sv.init(ns, n - 1);
      for (size_t i = 0; i < ns; i++) {
        const uint32_t p = cnt[size_t(e[i])]++;
        c.se[p] = e[i];
        c.sv.set(p, uint32_t(nidv[l.sv[i]]));
      }
      return;
    }

    c.ss.resize(ns);
    c.se.resize(ns);
    c.sv.init(ns, n - 1);
    for (size_t i = 0; i < ns; i++) {
      c.ss[i] = R(l.ss[i] + 1);
      c.se[i] = R(l.se[i] + 1);
      c.sv.set(i, uint32_t(nidv[l.sv[i]]));
    }

    const int N = l.lv.size(), C = l.kw.size();
    vector<uint32_t>& lid = S.lid;
    lid.resize(N);
    for (int q = 0; q < N; q++) lid[q] = uint32_t(nidv[l.lv[q]]);

    vector<int>& idx = S.idx;
    idx.resize(C);
    if (C > 1 && double(C) * log2(double(C)) > 3.0 * double(N + C)) {
      vector<int>& cnt = S.cnt;
      vector<int>& tmp = S.tmp;
      cnt.assign(N + 1, 0);
      tmp.resize(C);
      for (int j = 0; j < C; j++) cnt[N - 1 - l.khi[j]]++;
      for (int s = 0, q = 0; q <= N; q++) {
        const int x = cnt[q];
        cnt[q] = s;
        s += x;
      }
      for (int j = 0; j < C; j++) tmp[cnt[N - 1 - l.khi[j]]++] = j;
      fill(cnt.begin(), cnt.end(), 0);
      for (int j = 0; j < C; j++) cnt[l.klo[j]]++;
      for (int s = 0, q = 0; q <= N; q++) {
        const int x = cnt[q];
        cnt[q] = s;
        s += x;
      }
      for (int k = 0; k < C; k++) {
        const int j = tmp[k];
        idx[cnt[l.klo[j]]++] = j;
      }
    } else {
      iota(idx.begin(), idx.end(), 0);
      sort(idx.begin(), idx.end(), [&](int a, int b) {
        return l.klo[a] < l.klo[b] || (l.klo[a] == l.klo[b] && l.khi[a] > l.khi[b]);
      });
    }

    vector<int>& nlo = S.nlo;
    vector<int>& nhi = S.nhi;
    vector<int>& par = S.par;
    nlo.resize(C + 1);
    nhi.resize(C + 1);
    par.resize(C + 1);
    nlo[0] = 0;
    nhi[0] = N - 1;
    for (int k = 0; k < C; k++) {
      nlo[k + 1] = l.klo[idx[k]];
      nhi[k + 1] = l.khi[idx[k]];
    }

    {
      vector<int>& stk = S.stk;
      stk.assign(1, 0);
      for (int z = 1; z <= C; z++) {
        const int a = nlo[z], b = nhi[z];
        while (stk.size() > 1 && !(nlo[stk.back()] <= a && b <= nhi[stk.back()])) stk.pop_back();
        par[z] = stk.back();
        stk.push_back(z);
      }
    }

    vector<int>& chOff = S.chOff;
    vector<int>& chList = S.chList;
    chOff.assign(C + 2, 0);
    chList.resize(C);
    for (int z = 1; z <= C; z++) chOff[par[z] + 1]++;
    for (int z = 0; z <= C; z++) chOff[z + 1] += chOff[z];
    {
      vector<int>& pos = S.pos;
      pos.assign(chOff.begin(), chOff.end() - 1);
      for (int z = 1; z <= C; z++) chList[pos[par[z]]++] = z;
    }

    vector<int64_t>& itOff = S.itOff;
    itOff.assign(C + 2, 0);
    for (int z = 0; z <= C; z++) {
      const int ci = chOff[z], ce = chOff[z + 1];
      int64_t span = nhi[z] - nlo[z] + 1;
      for (int k = ci; k < ce; k++) span -= nhi[chList[k]] - nlo[chList[k]];
      itOff[z + 1] = span;
    }
    for (int z = 0; z <= C; z++) itOff[z + 1] += itOff[z];

    vector<pair<uint32_t, int>>& items = S.items;
    items.resize(itOff[C + 1]);
    vector<uint32_t>& minp = S.minp;
    minp.resize(C + 1);
    for (int z = C; z >= 0; z--) {
      int q = nlo[z], ci = chOff[z];
      const int ce = chOff[z + 1], h = nhi[z];
      int64_t itPos = itOff[z];
      bool sorted = true;
      uint32_t last = 0;
      while (q <= h) {
        uint32_t key;
        if (ci < ce && nlo[chList[ci]] == q) {
          const int ch = chList[ci];
          key = minp[ch];
          items[itPos++] = {key, ch};
          q = nhi[ch] + 1;
          ci++;
        } else {
          key = lid[q];
          items[itPos++] = {key, ~q};
          q++;
        }
        if (key < last) sorted = false;
        last = key;
      }
      auto b0 = items.begin() + itOff[z], b1 = items.begin() + itOff[z + 1];
      if (!sorted) {
        const size_t k = size_t(b1 - b0);
        if (!(k >= size_t(kEncRadix) && radix_items(&*b0, k, S.itmp))) sort(b0, b1);
      }
      minp[z] = b0 != b1 ? b0->first : UINT32_MAX;
    }

    vector<int>& leafOrder = S.order;
    vector<int>& newlo = S.newlo;
    newlo.resize(C + 1);
    leafOrder.resize(N);
    newlo[0] = 0;
    int* leafOrderData = leafOrder.data();
    for (int z = 0; z <= C; z++) {
      int pos = newlo[z];
      for (int64_t itPos = itOff[z], itEnd = itOff[z + 1]; itPos < itEnd; itPos++) {
        const auto it = items[itPos];
        if (it.second >= 0) {
          newlo[it.second] = pos;
          pos += nhi[it.second] - nlo[it.second] + 1;
        } else leafOrderData[pos++] = int(it.first);
      }
    }

    vector<int>& nodeOf = S.nodeOf;
    nodeOf.resize(C);
    for (int k = 0; k < C; k++) nodeOf[idx[k]] = k + 1;
    c.kw.resize(C);
    c.kpw.resize(C);
    for (int j = 0; j < C; j++) {
      c.kw[j] = R(l.kw[j] + 1);
      c.kpw[j] = R(l.kpw[j] + 1);
    }

    const int B = 1 << lfb;
    const int nb = (N + B - 1) / B;
    c.nlf = uint32_t(N);
    c.fd.resize(nb);
    uint64_t tot = 0;
    for (int k = 0; k < nb; k++) {
      const int p0 = k * B, p1 = min(N, p0 + B);
      int64_t mn = INT64_MAX, mx = INT64_MIN;
      for (int p = p0; p < p1; p++) {
        const int64_t y = int64_t(leafOrder[p]) - (p - p0);
        mn = min(mn, y);
        mx = max(mx, y);
      }
      const int w = bitsFor(uint64_t(mx - mn));
      c.fd[k] = {int32_t(mn), uint32_t(tot), uint8_t(w), 0, 0, 0};
      tot += (uint64_t(p1 - p0) * w + 7) / 8;
    }
    c.fst.assign(tot + 8, 0);
    for (int k = 0; k < nb; k++) {
      const int p0 = k * B, p1 = min(N, p0 + B), w = c.fd[k].w;
      uint8_t* bp = c.fst.data() + c.fd[k].off;
      uint64_t bit = 0;
      for (int p = p0; p < p1; p++, bit += w) {
        const uint64_t x = uint64_t(int64_t(leafOrder[p]) - (p - p0) - c.fd[k].base) & ((uint64_t(1) << w) - 1);
        uint64_t y = load8(bp + (bit >> 3));
        y |= x << (bit & 7);
        memcpy(bp + (bit >> 3), &y, 8);
      }
    }

    uint32_t maxCount = 0;
    for (int j = 0; j < C; j++) maxCount = max(maxCount, uint32_t(l.khi[j] - l.klo[j] + 1));
    c.ks.init(C, bitsFor(uint64_t(max(1, N - 1))));
    c.kc.init(C, bitsFor(maxCount));
    for (int j = 0; j < C; j++) {
      c.ks.set(j, uint64_t(newlo[nodeOf[j]]));
      c.kc.set(j, uint64_t(l.khi[j] - l.klo[j] + 1));
    }

    code_fill(c.lql, uint32_t(N), mu, [&](uint32_t p) { return uint32_t(leafOrder[p]); });
    code_fill(c.lqs, ns, mu, [&](uint32_t i) { return uint32_t(c.sv.get(i)); });
    key_fill(c.lkl, uint32_t(N), mu, [&](uint32_t p) { return uint32_t(leafOrder[p]); });
    key_fill(c.lks, ns, mu, [&](uint32_t i) { return uint32_t(c.sv.get(i)); });
  }

  vector<int> lMinMu;
  void prepare_lcodes(const Graph& G, const Borders& D) override;
  void finish_level(int mu, BlockOut& o) override;
  void save_tail(BlockOut& o) const override;
  void load(BlockIn& in, const Graph& G, uint64_t end) override;
  template <class F>
  void code_fill(BitArr& a, size_t len, int mu, F id_of) const {
    a.init(len, 1);
    uint8_t* out = a.b.data();
    uint64_t acc = 0;
    int na = 0;
    for (size_t i = 0; i < len; i++) {
      const uint64_t c = lMinMu[id_of(uint32_t(i))] <= mu;
      acc |= c << na;
      na++;
      if (na >= 64) {
        memcpy(out, &acc, 8);
        out += 8;
        na = 0;
        acc = 0;
      }
    }
    if (na > 0) memcpy(out, &acc, 8);
  }
  vector<uint8_t> lKeyMax;
  vector<uint32_t> lKeyEnd;
  const Borders* lBorders = nullptr;
  const Graph* lGraph = nullptr;
  uint8_t lkey(uint32_t u, int mu);
  template <class F>
  void key_fill(vector<uint8_t>& a, size_t len, int mu, F id_of) {
    if (mu < 3) return;
    a.resize(len);
    for (size_t i = 0; i < len; i++) a[i] = lkey(id_of(uint32_t(i)), mu);
  }
  void release_lcodes() override {
    vector<int>().swap(lMinMu);
    vector<uint8_t>().swap(lKeyMax);
    vector<uint32_t>().swap(lKeyEnd);
  }

  size_t core_phase(int epsRank, int mu, int* label, vector<pair<int, int>>& out, vector<int>& lcore,
                    int qk) const override {
    lcore.clear();
    const int T1 = epsRank + 1;
    const Lvl& lev = levels[mu];
    const bool useKey = qk > int(kmin[mu]);
    for (size_t i = 0, e = lev.ss.size(); i < e && int(lev.ss[i]) >= T1; i++) {
      if (int(lev.se[i]) >= T1) continue;
      const uint32_t v = uint32_t(lev.sv.get(i));
      const int origV = og(v);
      label[v] = origV;
      out.push_back({origV, origV});
      if (lev.lqs.get(i) && (!useKey || int(lev.lks[i]) >= qk)) lcore.push_back(int(v));
    }
    const uint8_t* qb = lev.lql.b.data();
    if (useKey) leaf_phase<true>(lev, T1, label, out, lcore, qb, qk);
    else leaf_phase<false>(lev, T1, label, out, lcore, qb, qk);
    return out.size();
  }

  template <bool kUseKey>
  __attribute__((noinline)) void leaf_phase(const Lvl& lk, int T1, int* label, vector<pair<int, int>>& out,
                                            vector<int>& lcore, const uint8_t* qb, int qk) const {
    const uint32_t bmask = (uint32_t(1) << lfb) - 1;
    const uint8_t* fst = lk.fst.data();
    const uint8_t* kb = kUseKey ? lk.lkl.data() : nullptr;
    constexpr int CB = 16;
    uint32_t cbs[CB], cbe[CB], cbf[CB];
    const typename Lvl::FD* fdp = lk.fd.data();
    const uint8_t* ogb = orig.b.data();
    const int ogw = orig.w;
    for (size_t j = 0, e = lk.kw.size();;) {
      int ncb = 0;
      for (; j < e && ncb < CB && int(lk.kw[j]) >= T1; j++) {
        if (int(lk.kpw[j]) >= T1) continue;
        const uint32_t p0 = uint32_t(lk.ks.get(j));
        cbs[ncb] = p0;
        cbe[ncb] = p0 + uint32_t(lk.kc.get(j));
        _mm_prefetch((const char*)&fdp[p0 >> lfb], _MM_HINT_T0);
        ncb++;
      }
      if (ncb == 0) break;
      for (int b = 0; b < ncb; b++) {
        const typename Lvl::FD& d = fdp[cbs[b] >> lfb];
        _mm_prefetch((const char*)(fst + d.off + ((uint64_t(cbs[b] & bmask) * d.w) >> 3)), _MM_HINT_T0);
      }
      for (int b = 0; b < ncb; b++) {
        cbf[b] = leaf(lk, cbs[b]);
        _mm_prefetch((const char*)(ogb + size_t(cbf[b]) * ogw), _MM_HINT_T0);
      }
      for (int b = 0; b < ncb; b++) {
        uint32_t p = cbs[b];
        const uint32_t pe = cbe[b];
        const int lb = og(cbf[b]);
        while (p < pe) {
          const uint32_t k = p >> lfb, bend = min(pe, (k + 1) << lfb);
          const typename Lvl::FD d = lk.fd[k];
          const unsigned w = d.w;
          const int32_t base = d.base - int32_t(k << lfb);
          if (w == 0) {
            for (; p < bend; p++) {
              const uint32_t v = uint32_t(base + int32_t(p));
              label[v] = lb;
              out.push_back({og(v), lb});
              if (((load8(qb + (p >> 3)) >> (p & 7)) & 1) && (!kUseKey || int(kb[p]) >= qk))
                lcore.push_back(int(v));
            }
            continue;
          }
          const uint8_t* bp = fst + d.off;
          const uint64_t msk = (uint64_t(1) << w) - 1;
          uint64_t bit = uint64_t(p & bmask) * w;
          for (; p < bend; p++, bit += w) {
            const uint32_t v = uint32_t(base + int32_t(p) + int32_t((load8(bp + (bit >> 3)) >> (bit & 7)) & msk));
            label[v] = lb;
            out.push_back({og(v), lb});
            if (((load8(qb + (p >> 3)) >> (p & 7)) & 1) && (!kUseKey || int(kb[p]) >= qk))
              lcore.push_back(int(v));
          }
        }
      }
    }
  }
};

static unique_ptr<CoreBase> new_compact_core(const Graph& G) {
  if (G.K <= 254) return unique_ptr<CoreBase>(new CompactCore<uint8_t>());
  if (G.K <= 65534) return unique_ptr<CoreBase>(new CompactCore<uint16_t>());
  return unique_ptr<CoreBase>(new CompactCore<uint32_t>());
}
