#pragma once
#include "borders.hpp"

inline void Borders::buildH(const Graph& G, const int* nid) {
  const int n = G.n;
  hi = max(2, G.Delta + 1);
  const int NL = hi + 3;
  const int* deg = G.deg.data();
  const int64_t* off = G.off.data();
  vector<int64_t> cap(NL + 1, 0), fill(NL + 1, 0);
  for (int x = 0; x < n; x++)
    if (deg[x] >= 2) cap[deg[x] + 1]++;
  for (int mu = NL - 1; mu >= 3; mu--) cap[mu] += cap[mu + 1];
  struct CandHead { int wbMax, v; };
  vector<hvec<CandHead>> slot(NL);
  vector<CandHead*> sp(NL, nullptr);
  for (int mu = 3; mu <= hi; mu++) {
    slot[mu].resize(cap[mu]);
    sp[mu] = slot[mu].data();
  }

  bool lScan;
  {
    mt19937_64 sr(0x5eed1234ull);
    const int64_t M2 = off[n];
    const int* lam = G.lamv.data();
    long long ok = 0, tot = 0;
    for (int trial = 0; trial < 4096 && M2 > 0; trial++) {
      const int64_t i = int64_t(sr() % uint64_t(M2));
      const int u = int(upper_bound(off, off + n + 1, i) - off) - 1;
      const int rest = (lam[i] - 1) - int(i - off[u]);
      ok += rest <= 16;
      tot++;
    }
    lScan = double(ok) / double(tot) >= 0.9;
  }

  {
    const int DD = G.Delta + 2;
    vector<int> wbMaxBuf(DD), simMaxBuf(DD);
    int* wbMax = wbMaxBuf.data();
    int* simMax = simMaxBuf.data();
    const int* nbr = G.nbr.data();
    const int* nsr = G.nsr.data();
    const int* tw = G.twin.data();
    const int* lam = G.lamv.data();
    for (int x = 0; x < n; x++) {
      const int dx = deg[x];
      if (dx < 2) continue;
      const int64_t b = off[x];
      for (int j = 0; j < dx; j++) wbMax[j] = -1;
      for (int l = 0; l <= dx; l++) simMax[l] = -1;
      for (int k = 0; k < dx; k++) {
        if (k + 16 < dx) _mm_prefetch((const char*)&off[nbr[b + k + 16]], _MM_HINT_T0);
        if (lScan) {
          if (k + 8 < dx) _mm_prefetch((const char*)(nsr + off[nbr[b + k + 8]] + tw[b + k + 8]), _MM_HINT_T0);
          if (k + 4 < dx) {
            const int w = nbr[b + k + 4];
            const int64_t ow = off[w];
            const int dw = int(off[w + 1] - ow);
            const int tw4 = tw[b + k + 4], Dw = dw < dx ? dw : dx;
            const int* s0 = nsr + ow + tw4;
            for (int dist = 16, nl = min(Dw - tw4, 112); dist < nl; dist += 16)
              _mm_prefetch((const char*)(s0 + dist), _MM_HINT_T0);
          }
        } else {
          if (k + 8 < dx) _mm_prefetch((const char*)&lam[off[nbr[b + k + 8]] + tw[b + k + 8]], _MM_HINT_T0);
          if (k + 4 < dx) {
            const int w = nbr[b + k + 4];
            const int64_t ow = off[w];
            const int dw = int(off[w + 1] - ow);
            const int lw = lam[ow + tw[b + k + 4]] - 1, Dw = dw < dx ? dw : dx;
            const int* s0 = nsr + ow + lw;
            for (int dist = 0, nl = min(Dw - lw, 96); dist < nl; dist += 16)
              _mm_prefetch((const char*)(s0 + dist), _MM_HINT_T0);
          }
        }
        const int u = nbr[b + k], sim = nsr[b + k];
        const int64_t ou = off[u];
        const int du = int(off[u + 1] - ou);
        const int D = du < dx ? du : dx;
        int lu;
        if (lScan) {
          const int tw0 = tw[b + k], jEnd = min(du, tw0 + 17);
          const int* su = nsr + ou;
          int j = tw0 + 1;
          while (j < jEnd && su[j] == sim) j++;
          lu = (j < jEnd || j == du) ? j : lam[ou + tw0] - 1;
        } else {
          lu = lam[ou + tw[b + k]] - 1;
        }
        const int l = lu < D ? lu : D;
        if (sim > simMax[l]) simMax[l] = sim;
        if (lu < D) {
          const int* su = nsr + ou;
          int j = lu;
          for (; j + 8 <= D; j += 8) {
            const __m256i suVec = _mm256_loadu_si256((const __m256i*)(su + j));
            const __m256i wbMaxVec = _mm256_loadu_si256((const __m256i*)(wbMax + j));
            _mm256_storeu_si256((__m256i*)(wbMax + j), _mm256_max_epi32(suVec, wbMaxVec));
          }
          for (; j < D; j++) wbMax[j] = max(wbMax[j], su[j]);
        }
      }
      const int v = nid[x];
      int M = -1;
      for (int j = dx - 1; j >= 1; j--) {
        M = max(M, simMax[j + 1]);
        wbMax[j] = max(wbMax[j], M);
        const int tauX = nsr[b + j];
        if (wbMax[j] > tauX) {
          const int mu = j + 2;
          sp[mu][fill[mu]++] = {wbMax[j], v};
        }
      }
    }
  }

  candOff.assign(NL, 0);
  int64_t s = 0;
  for (int mu = 0; mu + 1 < NL; mu++) {
    candOff[mu] = s;
    s += fill[mu];
  }
  candOff[NL - 1] = s;
  vBits = bitsFor(uint64_t(max(1, n - 1)));
  candRecBytes.assign(NL, uint8_t((vBits + 7) / 8));

  {
    hvec<CandHead> tmp, tmp2;
    vector<int64_t> cnt;
    for (int mu = 2; mu <= hi; mu++) {
      const int64_t len = candOff[mu + 1] - candOff[mu];
      if (len == 0) continue;
      tmp.resize(len);
      memcpy(tmp.data(), sp[mu], size_t(len) * sizeof(CandHead));
      sort_desc_by_rank(tmp, tmp2, G.K, [](const CandHead& x) { return x.wbMax; }, cnt);
      memcpy(sp[mu], tmp.data(), size_t(len) * sizeof(CandHead));
    }
  }

  layout_cands();
  for (int mu = 2; mu <= hi; mu++) {
    const int64_t len = candOff[mu + 1] - candOff[mu];
    const CandHead* e = sp[mu];
    pack_cand_level(mu, len, [e](int64_t i) { return e[i].v; });
    hvec<CandHead>().swap(slot[mu]);
  }
}

static vector<vector<int>> krtLeaves, krtGapW;

static void keep_krt_gaps(int mu, const Level& l) {
  const int nl = int(l.lv.size()), nk = int(l.kw.size());
  vector<int>& lv = krtLeaves[mu];
  vector<int>& W = krtGapW[mu];
  lv = l.lv;
  W.assign(nl > 1 ? size_t(nl - 1) : 0, -1);
  vector<int> o1(nk), o2(nk), cnt(size_t(nl) + 1);
  const vector<int>& kw = l.kw;
  for (int j = 0; j < nk; j++) cnt[size_t(nl - 1 - l.khi[j]) + 1]++;
  for (int k = 0; k < nl; k++) cnt[k + 1] += cnt[k];
  for (int j = 0; j < nk; j++) o1[cnt[size_t(nl - 1 - l.khi[j])]++] = j;
  fill(cnt.begin(), cnt.end(), 0);
  for (int j = 0; j < nk; j++) cnt[size_t(l.klo[j]) + 1]++;
  for (int k = 0; k < nl; k++) cnt[k + 1] += cnt[k];
  for (int i = 0; i < nk; i++) {
    const int j = o1[i];
    o2[cnt[size_t(l.klo[j])]++] = j;
  }
  vector<int> st;
  int ptr = 0;
  for (int k = 0; k + 1 < nl; k++) {
    while (!st.empty() && l.khi[st.back()] <= k) st.pop_back();
    while (ptr < nk && l.klo[o2[ptr]] <= k) {
      st.push_back(o2[ptr]);
      ptr++;
    }
    W[k] = st.empty() ? -1 : kw[st.back()];
  }
}

struct GapRmq {
  const int* w = nullptr;
  int n = 0, nb = 0, width = 1, nmb = 0;
  vector<uint32_t> mk;
  vector<int> bm;
  vector<uint32_t> tb;
  vector<int> mv;
  void build(const int* w_, int n_) {
    w = w_;
    n = n_;
    nb = (n + 31) >> 5;
    if (int(mk.size()) < n) mk.resize(size_t(n));
    for (int b = 0; b < nb; b++) {
      const int bs = b << 5, be = min(n, bs + 32);
      uint32_t m = 0;
      for (int p = bs; p < be; p++) {
        const int x = w[p];
        while (m) {
          const int q = 31 - __builtin_clz(m);
          if (w[bs + q] >= x) m &= ~(uint32_t(1) << q);
          else break;
        }
        m |= uint32_t(1) << (p - bs);
        mk[p] = m;
      }
    }

    bm.resize(size_t(nb));
    for (int b = 0; b < nb; b++) {
      const int be = min(n, (b << 5) + 32);
      bm[size_t(b)] = w[(b << 5) + __builtin_ctz(mk[be - 1])];
    }

    width = spWidth(nb);
    nmb = int((int64_t(nb) + width - 1) / width);
    const int rows = 32 - __builtin_clz(unsigned(nmb));
    tb.resize(size_t(nb));
    mv.resize(size_t(spRow(nmb, rows)));
    for (int block = 0; block < nmb; block++) {
      const int begin = block * width, end = begin + min(width, nb - begin);
      uint32_t m = 0;
      for (int i = begin; i < end; i++) {
        while (m) {
          const int q = begin + 31 - __builtin_clz(m);
          if (!(bm[size_t(i)] < bm[size_t(q)])) break;
          m &= ~(uint32_t(1) << (q - begin));
        }
        m |= uint32_t(1) << (i - begin);
        tb[size_t(i)] = m;
      }
      mv[size_t(block)] = bm[size_t(begin + __builtin_ctz(m))];
    }

    for (int j = 1; j < rows; j++) {
      int* row = mv.data() + spRow(nmb, j);
      const int* prev = mv.data() + spRow(nmb, j - 1);
      const int half = 1 << (j - 1), count = nmb - (1 << j) + 1;
      for (int i = 0; i < count; i++) row[i] = min(prev[i], prev[i + half]);
    }
  }
  int inb(int l, int r) const {
    const int bs = l & ~31;
    return w[bs + __builtin_ctz(mk[r] & (~0u << (l - bs)))];
  }
  int inm(int l, int r) const { return bm[size_t(spInside(tb.data(), width, l, r))]; }
  int query(int l, int r) const {
    const int bl = l >> 5, br = r >> 5;
    if (bl == br) return inb(l, r);
    int wMin = min(inb(l, (bl << 5) + 31), inb(br << 5, r));
    if (br - bl > 1) {
      const int a = bl + 1, b = br - 1, ma = a / width, mb = b / width;
      if (ma == mb) return min(wMin, inm(a, b));
      wMin = min(wMin, min(inm(a, (ma + 1) * width - 1), inm(mb * width, b)));
      const int first = ma + 1, last = mb - 1;
      if (first <= last) {
        const int j = 31 - __builtin_clz(unsigned(last - first + 1));
        const int* row = mv.data() + spRow(nmb, j);
        wMin = min(wMin, min(row[first], row[last - (1 << j) + 1]));
      }
    }
    return wMin;
  }
  int bottleneck(int a, int b) const { return query(a, b - 1); }
};

struct BHRec { int wb, lo, x; uint32_t u; };
template <class T>
inline void permute_in_place(T* a, uint32_t* order, uint32_t d) {
  for (uint32_t s = 0; s < d; s++) {
    if (order[s] == s) continue;
    const T t = a[s];
    uint32_t q = s;
    while (order[q] != s) {
      const uint32_t nx = order[q];
      a[q] = a[nx];
      order[q] = q;
      q = nx;
    }
    a[q] = t;
    order[q] = q;
  }
}
inline void order_by_wb(const BHRec* BHmu, uint32_t sizeBHmu, uint32_t* order) {
  if (sizeBHmu <= 32) {
    for (uint32_t i = 0; i < sizeBHmu; i++) {
      uint32_t k = i;
      while (k > 0 && BHmu[order[k - 1]].wb < BHmu[i].wb) {
        order[k] = order[k - 1];
        k--;
      }
      order[k] = i;
    }
    return;
  }

  int wbMin = BHmu[0].wb, wbMax = BHmu[0].wb;
  for (uint32_t i = 1; i < sizeBHmu; i++) {
    wbMin = min(wbMin, BHmu[i].wb);
    wbMax = max(wbMax, BHmu[i].wb);
  }
  const uint64_t range = uint64_t(int64_t(wbMax) - int64_t(wbMin)) + 1;
  if (range <= 4 * uint64_t(sizeBHmu) + 64) {
    vector<uint32_t> cnt(static_cast<size_t>(range), 0);
    for (uint32_t i = 0; i < sizeBHmu; i++) cnt[size_t(int64_t(wbMax) - BHmu[i].wb)]++;
    uint64_t acc = 0;
    for (uint32_t& c : cnt) {
      const uint64_t h = c;
      c = uint32_t(acc);
      acc += h;
    }
    for (uint32_t i = 0; i < sizeBHmu; i++) order[cnt[size_t(int64_t(wbMax) - BHmu[i].wb)]++] = i;
    return;
  }

  iota(order, order + sizeBHmu, uint32_t(0));
  sort(order, order + sizeBHmu, [&](uint32_t a, uint32_t b) {
    return BHmu[a].wb != BHmu[b].wb ? BHmu[a].wb > BHmu[b].wb : a < b;
  });
}
template <bool kFast>
inline void build_bh_route(Graph& G, Borders& D, const CoreBase& C, const int* nid, const int* oid,
                           BlockOut& o, vector<int64_t>& sizeBH) {
  const int n = G.n, K = G.K;
  const int64_t* off = G.off.data();
  const int* nbr = G.nbr.data();
  const int* nsr = G.nsr.data();
  const int* lam = G.lamv.data();
  const int* tw = G.twin.data();
  const int muMax = G.Delta + 1;
  int nbK = 0;
  while ((int64_t(1) << nbK) < int64_t(K)) nbK++;
  const int kcap = 64, gsmall = 48;
  constexpr int BB = kFast ? 16 : 1;
  const size_t BATCH = 1048576;
  const size_t PF = 8;

  struct ActiveEntry { uint32_t p, ou; int x, lo, fhi, hi; };
  struct TI { uint32_t u; int wb, pos; uint32_t kk; };
  struct IT { uint32_t u; int wb, pos; uint32_t kf; };
  struct GR {
    int li;
    uint32_t fa, fb, ta, tb, ia, il;
    uint32_t hk;
    int hpos;
    uint32_t ha, hb;
  };
  struct GM { int li; uint32_t ks, kc; };
  struct FI { uint32_t u; int wb, pos, fg; };
  struct LvB {
    int nlv = 0;
    int64_t len = 0;
    long long an = 0;
    vector<TI> Fv, Tv;
    vector<GR> grs;
    vector<GM> gms;
    vector<Borders::KeptEntry> kept;
    vector<FI> fbv;
    vector<int> fbS, fbG, lx;
    GapRmq rmq;
    vector<uint32_t> hkv;
  };
  vector<LvB> lb(static_cast<size_t>(BB));
  struct ActiveRef { uint32_t p; int x; };
  using Active = typename conditional<kFast, ActiveEntry, ActiveRef>::type;
  vector<Active> act, act2;

  const auto entry = [&](uint32_t p32, int x) {
    const int64_t p = p32, ou = off[nbr[p]];
    const int dx = int(off[x + 1] - off[x]);
    const int du = int(off[nbr[p] + 1] - ou);
    return ActiveEntry{p32, uint32_t(ou), x, lam[p] + 1,
                       min(lam[ou + tw[p]], dx + 1), min(dx, du) + 1};
  };
  const auto unpack = [&](const Active& a) -> ActiveEntry {
    if constexpr (kFast) return a;
    else return entry(a.p, a.x);
  };

  size_t actN = 0;
  vector<TI> Ttmp;
  vector<IT> Iv;
  vector<int> kpos, kidx, kgi, ordp, lnkP, lnkN, lastG, giv;
  vector<char> keepv;
  vector<int64_t> cb;
  vector<BHRec> BHmu;
  vector<uint32_t> order;
  const size_t kKeep = size_t(64) << 20;
  sizeBH.assign(size_t(muMax) + 1, 0);
  vector<int> tauX(size_t(BB), 0);
  vector<uint32_t> fa0(size_t(BB), 0), ta0(size_t(BB), 0);
  vector<uint64_t> dead(size_t(off[n] + 63) / 64 + 1, 0);
  const auto is_dead = [&](uint32_t kk) { return (dead[kk >> 6] >> (kk & 63)) & 1u; };
  const auto t_first = [](const TI* t, const TI* f) { return t->wb > f->wb || (t->wb == f->wb && t->kk < f->kk); };
  vector<uint32_t> pend;

  const size_t nB = size_t(n) * size_t(BB), nBy = ((nB * sizeof(int) + 63) / 64) * 64 + 64;
  int* posB = static_cast<int*>(aligned_alloc(64, nBy));
  int* lmapB = static_cast<int*>(aligned_alloc(64, nBy));
  if (!posB || !lmapB) die("cannot allocate the one-layer position maps");
  fill(posB, posB + nB, -1);

  const int nBlk = muMax >= 3 ? (muMax - 3) / BB + 1 : 0;
  vector<int64_t> bkS(size_t(nBlk) + 1, 0);
  vector<uint32_t> bkt;
  {
    const auto block_of = [&](int x, int64_t p) {
      const int dx = int(off[x + 1] - off[x]);
      const int u = nbr[p], du = int(off[u + 1] - off[u]);
      const int lo = lam[p] + 1, hi = min(dx, du) + 1;
      return hi >= lo ? (hi - 3) / BB : -1;
    };
    for (int x = 0; x < n; x++) {
      for (int64_t p = off[x]; p < off[x + 1]; p++) {
        const int bi = block_of(x, p);
        if (bi >= 0) bkS[size_t(bi) + 1]++;
      }
    }
    for (int bi = 0; bi < nBlk; bi++) bkS[size_t(bi) + 1] += bkS[size_t(bi)];
    bkt.resize(size_t(bkS[size_t(nBlk)]));
    vector<int64_t> cur(bkS.begin(), bkS.end() - 1);
    for (int x = 0; x < n; x++) {
      for (int64_t p = off[x]; p < off[x + 1]; p++) {
        const int bi = block_of(x, p);
        if (bi >= 0) bkt[size_t(cur[size_t(bi)]++)] = uint32_t(p);
      }
    }
  }

  for (int bi = nBlk - 1; bi >= 0; bi--) {
    const int m0 = 3 + bi * BB, m1 = min(muMax, m0 + BB - 1), nmu = m1 - m0 + 1;
    size_t na = 0;
    for (size_t i = 0; i < actN; i++) {
      int lo;
      if constexpr (kFast) lo = act[i].lo;
      else lo = lam[act[i].p] + 1;
      if (lo <= m1) act[na++] = act[i];
    }
    const size_t N = na + size_t(bkS[size_t(bi) + 1] - bkS[size_t(bi)]);
    if constexpr (kFast) {
      if (N > act2.capacity()) {
        vector<Active>().swap(act2);
        act2.reserve(N);
      }
    }
    act2.resize(N);

    size_t ia = 0, wa = 0;
    int owner = 0;
    for (int64_t i = bkS[size_t(bi)]; i < bkS[size_t(bi) + 1]; i++) {
      const uint32_t p = bkt[size_t(i)];
      if (off[owner + 1] <= p)
        owner = int(upper_bound(off + owner + 1, off + n + 1, int64_t(p)) - off) - 1;
      while (ia < na && act[ia].p < p) act2[wa++] = act[ia++];
      if constexpr (kFast) act2[wa++] = entry(p, owner);
      else act2[wa++] = {p, owner};
    }
    while (ia < na) act2[wa++] = act[ia++];
    act.swap(act2);
    actN = act.size();

    if constexpr (kFast) {
      long long dcnt[BB + 1] = {};
      for (size_t k = 0; k < actN; k++) {
        const ActiveEntry& e = act[k];
        dcnt[size_t(max(e.lo, m0) - m0)]++;
        dcnt[size_t(min(e.hi, m1) - m0 + 1)]--;
      }
      long long run = 0;
      for (int j = 0; j < nmu; j++) {
        run += dcnt[size_t(j)];
        lb[size_t(j)].an = run;
      }
    } else {
      lb[0].an = (long long)actN;
    }

    for (int j = 0; j < nmu; j++) {
      LvB& lvb = lb[size_t(j)];
      const int mu = m0 + j;
      lvb.len = D.candOff[mu + 1] - D.candOff[mu];
      lvb.gms.clear();
      lvb.nlv = 0;
      if (lvb.an == 0) {
        vector<int>().swap(krtLeaves[mu]);
        vector<int>().swap(krtGapW[mu]);
        continue;
      }
      const vector<int>& lv = krtLeaves[mu];
      const vector<int>& W = krtGapW[mu];
      lvb.nlv = int(lv.size());
      for (int q = 0; q < lvb.nlv; q++) posB[size_t(lv[q]) * BB + j] = q;
      if (lvb.nlv >= 2) lvb.rmq.build(W.data(), int(W.size()));
      lvb.lx.resize(size_t(lvb.len));
      for (int64_t i = 0; i < lvb.len; i++) {
        const uint32_t v = D.cand_vertex(mu, i);
        const int x = oid[v];
        int& lm = lmapB[size_t(x) * BB + j];
        lm = int(i);
      }
    }

    for (size_t ka = 0; ka < actN;) {
      size_t ke = ka + min(BATCH, actN - ka);
      while (ke < actN && act[ke].x == act[ke - 1].x) ke++;
      for (int j = 0; j < nmu; j++) {
        LvB& lvb = lb[size_t(j)];
        lvb.Fv.clear();
        lvb.Tv.clear();
        lvb.grs.clear();
      }
      for (size_t k = ka; k < ke;) {
        const int x = act[k].x;
        const int64_t bx = off[x];
        const int dx = int(off[x + 1] - bx);
        const int jx = min(m1, dx + 1) - m0;
        for (int j = 0; j <= jx; j++) {
          tauX[size_t(j)] = nsr[bx + m0 + j - 2];
          fa0[size_t(j)] = uint32_t(lb[size_t(j)].Fv.size());
          ta0[size_t(j)] = uint32_t(lb[size_t(j)].Tv.size());
        }
        for (; k < ke && act[k].x == x; k++) {
          if (k + PF < ke) {
            const ActiveEntry ahead = unpack(act[k + PF]);
            __builtin_prefetch(posB + size_t(nbr[ahead.p]) * BB);
            const int t0 = max(max(ahead.lo, m0), ahead.fhi + 1), t1 = min(ahead.hi, m1);
            if (t0 <= t1) {
              __builtin_prefetch(nsr + ahead.ou + t0 - 2);
              __builtin_prefetch(nsr + ahead.ou + t1 - 2);
            }
          }
          const ActiveEntry e = unpack(act[k]);
          const int muLo = max(e.lo, m0), muHi = min(e.hi, m1);
          const uint32_t u = nbr[e.p];
          const uint32_t kk = e.p;
          const int* pu = posB + size_t(u) * BB;
          const int f1 = min(muHi, e.fhi);
          const int t0 = max(muLo, e.fhi + 1);
          if (is_dead(kk)) continue;
          if (muLo <= f1) {
            const int sim = nsr[e.p];
            for (int mu = muLo; mu <= f1; mu++) lb[size_t(mu - m0)].Fv.push_back({u, sim, pu[mu - m0], kk});
          }
          if (t0 <= muHi) {
            const int* su = nsr + e.ou;
            for (int mu = t0; mu <= muHi; mu++) {
              LvB& lvb = lb[size_t(mu - m0)];
              const int tauU = su[mu - 2];
              if (tauU > tauX[size_t(mu - m0)]) lvb.Tv.push_back({u, tauU, pu[mu - m0], kk});
            }
          }
        }
        for (int j = 0; j <= jx; j++) {
          LvB& lvb = lb[size_t(j)];
          if (lvb.Fv.size() > fa0[size_t(j)] || lvb.Tv.size() > ta0[size_t(j)]) {
            const int li = lmapB[size_t(x) * BB + j];
            lvb.grs.push_back({li, fa0[size_t(j)], uint32_t(lvb.Fv.size()),
                               ta0[size_t(j)], uint32_t(lvb.Tv.size()), 0, 0});
          }
        }
      }

      for (int jj = 0; jj < nmu; jj++) {
        const int j = nmu - 1 - jj;
        LvB& lvb = lb[size_t(j)];
        if (lvb.grs.empty()) continue;
        if (Iv.size() < lvb.Fv.size() + lvb.Tv.size()) Iv.resize(lvb.Fv.size() + lvb.Tv.size());
        uint32_t nIv = 0;
        lvb.hkv.clear();
        for (GR& g : lvb.grs) {
          auto hcov = [&](int ps, int wb) {
            const int pa = min(g.hpos, ps), pb = max(g.hpos, ps);
            return lvb.rmq.bottleneck(pa, pb) >= wb;
          };
          g.ha = g.hb = uint32_t(lvb.hkv.size());
          {
            const uint32_t fEnd = g.fb, tEnd = g.tb;
            const long long nFT = (long long)(fEnd - g.fa) + (long long)(tEnd - g.ta);
            uint32_t fi = g.fa;
            while (fi < fEnd && is_dead(lvb.Fv[fi].kk)) fi++;
            const TI* hT = nullptr;
            for (uint32_t i = g.ta; i < tEnd; i++)
              if (!hT || lvb.Tv[i].wb > hT->wb) hT = &lvb.Tv[i];
            const TI* hF = fi < fEnd ? &lvb.Fv[fi] : nullptr;
            const TI* h = hF ? hF : hT;
            if (hF && hT) {
              h = t_first(hT, hF) ? hT : hF;
            }
            if (h) {
              g.hk = h->kk;
              g.hpos = h->pos;
            }
            if (h && nFT >= 2) {
              if (g.hpos >= 0) {
                uint32_t keep = g.fa;
                for (uint32_t i = g.fa; i < fEnd; i++) {
                  const TI s = lvb.Fv[i];
                  if (s.kk != g.hk && s.pos >= 0) {
                    if (!is_dead(s.kk) && hcov(s.pos, s.wb)) {
                      lvb.hkv.push_back(s.kk);
                      continue;
                    }
                  }
                  lvb.Fv[keep++] = s;
                }
                g.fb = keep;
                keep = g.ta;
                for (uint32_t i = g.ta; i < tEnd; i++) {
                  const TI s = lvb.Tv[i];
                  if (s.kk != g.hk && s.pos >= 0 && hcov(s.pos, s.wb)) {
                    continue;
                  }
                  lvb.Tv[keep++] = s;
                }
                g.tb = keep;
                g.hb = uint32_t(lvb.hkv.size());
              }
            }
          }

          TI* t = lvb.Tv.data() + g.ta;
          const int nt = int(g.tb - g.ta);
          if (nt > 1) {
            if (nt <= gsmall) {
              for (int i = 1; i < nt; i++) {
                const TI s = t[i];
                int k = i;
                while (k > 0 && t[k - 1].wb < s.wb) {
                  t[k] = t[k - 1];
                  k--;
                }
                t[k] = s;
              }
            } else {
              if (int(Ttmp.size()) < nt) Ttmp.resize(size_t(nt));
              TI* src = t;
              TI* dst = Ttmp.data();
              for (int sh = 0; sh < nbK; sh += 8) {
                uint32_t cnt[257];
                memset(cnt, 0, sizeof cnt);
                for (int i = 0; i < nt; i++) cnt[((uint32_t(K - 1 - src[i].wb) >> sh) & 255u) + 1]++;
                bool one = false;
                for (int d = 1; d <= 256 && !one; d++) one = cnt[d] == uint32_t(nt);
                if (one) continue;
                for (int d = 0; d < 256; d++) cnt[d + 1] += cnt[d];
                for (int i = 0; i < nt; i++) dst[cnt[(uint32_t(K - 1 - src[i].wb) >> sh) & 255u]++] = src[i];
                swap(src, dst);
              }
              if (src != t) memcpy(t, src, size_t(nt) * sizeof(TI));
            }
          }

          g.ia = nIv;
          auto put = [&](const TI& s, const uint32_t isF) {
            if (isF && is_dead(s.kk)) return;
            Iv[nIv++] = {s.u, s.wb, s.pos, (s.kk << 1) | isF};
          };
          const TI* f = lvb.Fv.data() + g.fa;
          const TI* fe = lvb.Fv.data() + g.fb;
          const TI* te = lvb.Tv.data() + g.tb;
          const TI* tt = t;
          while (f < fe && tt < te) {
            if (t_first(tt, f)) put(*tt++, 0u);
            else put(*f++, 1u);
          }
          for (; f < fe; f++) put(*f, 1u);
          for (; tt < te; tt++) put(*tt, 0u);
          g.il = nIv - g.ia;
        }

        for (const GR& g : lvb.grs) {
          const IT* it = Iv.data() + g.ia;
          const int ng = int(g.il);
          bool fb = false;
          pend.clear();
          kpos.clear();
          kidx.clear();
          kgi.clear();
          kpos.push_back(it[0].pos);
          kidx.push_back(0);
          kgi.push_back(-1);
          for (int i = 1; i < ng; i++) {
            const int pi = it[i].pos, wi = it[i].wb, kc = int(kpos.size());
            int r = 0;
            while (r < kc && kpos[r] < pi) r++;
            int gi = -1;
            bool pruned = false;
            if (r > 0) {
              const int cv = lvb.rmq.bottleneck(kpos[r - 1], pi);
              if (cv >= wi) pruned = true;
              else gi = cv;
            }
            if (!pruned && r < kc) {
              const int cv = lvb.rmq.bottleneck(pi, kpos[r]);
              if (cv >= wi) pruned = true;
              else gi = max(gi, cv);
            }
            const uint32_t kf = it[i].kf;
            if (pruned) {
              if (kf & 1u) pend.push_back(kf >> 1);
              continue;
            }
            if (kc >= kcap) {
              fb = true;
              break;
            }
            kpos.insert(kpos.begin() + r, pi);
            kidx.push_back(i);
            kgi.push_back(gi);
          }
          if (!fb) {
            for (uint32_t h = g.ha; h < g.hb; h++) pend.push_back(lvb.hkv[h]);
            for (const uint32_t kk : pend) dead[kk >> 6] |= uint64_t(1) << (kk & 63);
            lvb.gms.push_back({g.li, uint32_t(lvb.kept.size()), uint32_t(kidx.size())});
            for (size_t ki = 0; ki < kidx.size(); ki++) {
              const IT& s = it[kidx[ki]];
              lvb.kept.push_back({uint32_t(nid[s.u]), s.wb, kgi[ki]});
            }
          } else {
            lvb.fbS.push_back(int(lvb.fbv.size()));
            lvb.fbG.push_back(int(lvb.gms.size()));
            lvb.gms.push_back({g.li, 0, 0});
            const int fg = int(lvb.fbS.size()) - 1;
            for (int i = 0; i < ng; i++) lvb.fbv.push_back({it[i].u, it[i].wb, it[i].pos, fg});
          }
        }
      }
      ka = ke;
    }

    for (int jj = 0; jj < nmu; jj++) {
      const int j = nmu - 1 - jj;
      LvB& lvb = lb[size_t(j)];
      const int mu = m0 + j;
      if (lvb.an == 0) continue;
      const int nlv = lvb.nlv;
      if (!lvb.fbS.empty()) {
        vector<FI>& fbv = lvb.fbv;
        vector<int>& fbS = lvb.fbS;
        const int nf = int(fbS.size()), nfbv = int(fbv.size());
        fbS.push_back(nfbv);
        cb.assign(size_t(nlv) + 1, 0);
        for (int k = 0; k < nfbv; k++) cb[size_t(fbv[k].pos) + 1]++;
        for (int q = 0; q < nlv; q++) cb[q + 1] += cb[q];
        ordp.resize(size_t(nfbv));
        for (int k = 0; k < nfbv; k++) ordp[cb[size_t(fbv[k].pos)]++] = k;
        lnkP.assign(size_t(nfbv), -1);
        lnkN.assign(size_t(nfbv), -1);
        lastG.assign(size_t(nf), -1);
        for (int r = 0; r < nfbv; r++) {
          const int k = ordp[r], q = fbv[k].fg;
          if (lastG[q] >= 0) {
            lnkN[lastG[q]] = k;
            lnkP[k] = lastG[q];
          }
          lastG[q] = k;
        }
        giv.resize(size_t(nfbv));
        keepv.resize(size_t(nfbv));
        for (int q = 0; q < nf; q++)
          for (int k = fbS[q + 1] - 1; k >= fbS[q]; k--) {
            const int pred = lnkP[k], succ = lnkN[k];
            int gi = -1;
            if (pred >= 0) gi = max(gi, lvb.rmq.bottleneck(fbv[pred].pos, fbv[k].pos));
            if (succ >= 0) gi = max(gi, lvb.rmq.bottleneck(fbv[k].pos, fbv[succ].pos));
            if (pred >= 0) lnkN[pred] = succ;
            if (succ >= 0) lnkP[succ] = pred;
            giv[k] = gi;
            keepv[k] = gi < fbv[k].wb;
          }
        for (int q = 0; q < nf; q++) {
          GM& m = lvb.gms[size_t(lvb.fbG[q])];
          m.ks = uint32_t(lvb.kept.size());
          for (int k = fbS[q]; k < fbS[q + 1]; k++)
            if (keepv[k]) lvb.kept.push_back({uint32_t(nid[fbv[k].u]), fbv[k].wb, giv[k]});
          m.kc = uint32_t(lvb.kept.size()) - m.ks;
        }
      }

      vector<FI>().swap(lvb.fbv);
      vector<int>().swap(lvb.fbS);
      vector<int>().swap(lvb.fbG);
      vector<int>().swap(ordp);
      vector<int>().swap(lnkP);
      vector<int>().swap(lnkN);
      vector<int>().swap(lastG);
      vector<int>().swap(giv);
      vector<char>().swap(keepv);
      vector<int64_t>().swap(cb);
      const vector<int>& lv = krtLeaves[mu];
      for (int q = 0; q < nlv; q++) posB[size_t(lv[q]) * BB + j] = -1;
      vector<int>().swap(krtLeaves[mu]);
      vector<int>().swap(krtGapW[mu]);

      if constexpr (kFast) {
        vector<int>& lx = lvb.lx;
        fill(lx.begin(), lx.end(), -1);
        for (size_t g = 0; g < lvb.gms.size(); g++) {
          const int li = lvb.gms[g].li;
          lx[size_t(li)] = int(g);
        }
      } else {
        vector<int>().swap(lvb.lx);
        sort(lvb.gms.begin(), lvb.gms.end(), [](const GM& a, const GM& b) { return a.li < b.li; });
      }
      const auto for_each_group = [&](auto fn) {
        if constexpr (kFast) {
          for (int64_t li = 0; li < lvb.len; li++)
            if (lvb.lx[size_t(li)] >= 0) fn(lvb.gms[size_t(lvb.lx[size_t(li)])]);
        } else {
          for (const GM& m : lvb.gms) fn(m);
        }
      };

      const int64_t sizeBHmu = int64_t(lvb.kept.size());
      sizeBH[mu] = sizeBHmu;
      if (!sizeBHmu) {
        vector<Borders::KeptEntry>().swap(lvb.kept);
        vector<int>().swap(lvb.lx);
        continue;
      }
      if (size_t(sizeBHmu) > BHmu.capacity()) {
        vector<BHRec>().swap(BHmu);
        BHmu.reserve(size_t(sizeBHmu));
      }
      BHmu.resize(size_t(sizeBHmu));
      {
        size_t q = 0;
        for_each_group([&](const GM& m) {
          if (!m.kc) return;
          const uint32_t v = D.cand_vertex(mu, m.li);
          const int origV = C.og(v), x = oid[v];
          const int64_t bx = off[x];
          const int tauXmu = nsr[bx + mu - 2];
          for (uint32_t k = m.ks; k < m.ks + m.kc; k++) {
            const Borders::KeptEntry& it = lvb.kept[k];
            BHmu[q++] = {it.wb, max(it.gi, tauXmu), origV, it.u};
          }
        });
      }
      vector<int>().swap(lvb.lx);
      vector<Borders::KeptEntry>().swap(lvb.kept);

      if (size_t(sizeBHmu) > order.capacity()) {
        vector<uint32_t>().swap(order);
        order.reserve(size_t(sizeBHmu));
      }
      order.resize(size_t(sizeBHmu));
      order_by_wb(BHmu.data(), uint32_t(sizeBHmu), order.data());
      permute_in_place(BHmu.data(), order.data(), uint32_t(sizeBHmu));
      if (order.capacity() * sizeof(uint32_t) > kKeep) vector<uint32_t>().swap(order);

      const int hb = Borders::HLists::hb;
      const int64_t nbl = (sizeBHmu + (int64_t(1) << hb) - 1) >> hb, sizeST = spSize(nbl);
      o.begin("h.lv", uint64_t(32 + 16 * sizeBHmu + 4 * nbl + 4 * sizeST));
      const uint64_t h[4] = {uint64_t(mu), uint64_t(sizeBHmu), uint64_t(nbl), uint64_t(sizeST)};
      o.put(h, sizeof h);
      o.field4(&BHmu[0].wb, sizeof(BHRec), size_t(sizeBHmu));
      o.field4(&BHmu[0].lo, sizeof(BHRec), size_t(sizeBHmu));
      o.field4(&BHmu[0].x, sizeof(BHRec), size_t(sizeBHmu));
      o.field4(&BHmu[0].u, sizeof(BHRec), size_t(sizeBHmu));
      vector<int> BM(static_cast<size_t>(nbl));
      for (int64_t b = 0; b < nbl; b++) {
        int mn = INT_MAX;
        for (int64_t k = b << hb, e = min(sizeBHmu, (b + 1) << hb); k < e; k++) mn = min(mn, BHmu[size_t(k)].lo);
        BM[size_t(b)] = mn;
      }
      if (BHmu.capacity() * sizeof(BHRec) > kKeep) vector<BHRec>().swap(BHmu);
      vector<uint32_t> ST(static_cast<size_t>(sizeST));
      spBuild<int, true>(BM.data(), int(nbl), ST.data());
      o.put(BM.data(), size_t(nbl) * 4);
      o.put(ST.data(), size_t(sizeST) * 4);
    }
  }

  free(posB);
  free(lmapB);

  G.release_border_arrays();
  vector<int>().swap(G.nbr);
  hvec<int>().swap(G.nsr);
}
inline void build_bh(Graph& G, Borders& D, const CoreBase& C, const int* nid, const int* oid,
                     BlockOut& o, vector<int64_t>& sizeBH, bool fast) {
  if (fast) build_bh_route<true>(G, D, C, nid, oid, o, sizeBH);
  else build_bh_route<false>(G, D, C, nid, oid, o, sizeBH);
}
