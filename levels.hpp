#pragma once
#include "graph.hpp"

struct Level {
  vector<int> ss, se, sv;
  vector<int> kw, kpw;
  vector<int> klo, khi, lv;
  void free() {
    vector<int>().swap(ss);
    vector<int>().swap(se);
    vector<int>().swap(sv);
    vector<int>().swap(kw);
    vector<int>().swap(kpw);
    vector<int>().swap(klo);
    vector<int>().swap(khi);
    vector<int>().swap(lv);
  }
};

static void build_levels(const Graph& G, const std::function<void(int, Level&)>& sink) {
  const int n = G.n, muMax = G.Delta + 1, K = G.K;
  const int64_t m = G.m;
  hvec<int> ord(n), nid(n);
  {
    vector<int> c(G.Delta + 2, 0);
    for (int v = 0; v < n; v++) c[G.Delta - G.deg[v]]++;
    for (int s = 0, k = 0; k <= G.Delta; k++) {
      int t = c[k];
      c[k] = s;
      s += t;
    }
    for (int v = 0; v < n; v++) {
      int x = c[G.Delta - G.deg[v]]++;
      ord[x] = v;
      nid[v] = x;
    }
  }

  vector<int> sizeV(muMax + 2, 0);
  {
    int sizeVmu = n;
    for (int mu = 1; mu <= muMax + 1; mu++) {
      while (sizeVmu > 0 && G.deg[ord[sizeVmu - 1]] < mu - 1) sizeVmu--;
      sizeV[mu] = sizeVmu;
    }
  }

  hvec<uint32_t> voff(n);
  for (int x = 0; x < n; x++) voff[x] = uint32_t(G.off[ord[x]]);
  hvec<int64_t> ooff(n + 1);
  ooff[0] = 0;
  {
    hvec<int> odg(n);
    for (int x = 0; x < n; x++) {
      const int origX = ord[x];
      int c = 0;
      for (int64_t p = G.off[origX]; p < G.off[origX + 1]; p++) c += nid[G.nbr[p]] < x;
      odg[x] = c;
    }
    for (int x = 0; x < n; x++) ooff[x + 1] = ooff[x] + odg[x];
  }

  hvec<SimEdge> rec(m);
  for (int x = 0; x < n; x++) {
    const int origX = ord[x];
    int64_t q = ooff[x];
    for (int64_t p = G.off[origX]; p < G.off[origX + 1]; p++) {
      const int y = nid[G.nbr[p]];
      if (y < x) rec[q++] = {x, y, G.nsr[p]};
    }
  }

  vector<int> e1(n, -1);

  {
    struct alignas(16) VertexState { int p, stamp, tau, up, tr, uw, fe; uint32_t off; };
    hvec<VertexState> vs(n);
    for (int x = 0; x < n; x++) vs[x] = {x, 0, -1, x, ~x, 0, 0, voff[x]};
    struct KrtNode { int w, pw, lc, rc, sz, off; };
    hvec<KrtNode> kn(n);
    struct CoreEdge { int w, sim, u, v; uint32_t ko; };
    hvec<CoreEdge> Fprev(n), Fmu(n);
    int64_t nPrev = 0;
    vector<vector<int>> bucket(muMax + 2);
    vector<CoreEdge> fresh, ftmp, excluded, woken;
    vector<int64_t> cnt;
    struct SingletonRec { int tau, x, e; };
    vector<SingletonRec> sc, sctmp;
    const int* nsr = G.nsr.data();
    int mu = muMax;
    int64_t wsteps = 0;
    auto bottleneck_walk = [&](int a, int b) {
      int w = -1;
      while (a != b) {
        const VertexState &va = vs[a], &vb = vs[b];
        if (va.up != a && (vb.up == b || va.tr < vb.tr)) {
          w = va.uw;
          a = va.up;
        } else {
          w = vb.uw;
          b = vb.up;
        }
        wsteps++;
      }
      return w;
    };

    hvec<int> rpos, rM;
    hvec<uint64_t> rmask;
    vector<int> rbm;
    vector<uint32_t> rtb;
    int rnb = 0;
    auto rmq_build = [&](int nc) {
      rpos.resize(n);
      rM.resize(n);
      rmask.resize(n);
      int nl = 0;
      for (int q = 0; q < nc; q++) {
        const KrtNode& z = kn[q];
        const int o = z.off;
        int c = z.lc, leftSize;
        if (c >= 0) leftSize = kn[c].sz;
        else {
          rpos[~c] = o;
          leftSize = 1;
        }
        rM[o + leftSize - 1] = z.w;
        c = z.rc;
        if (c < 0) rpos[~c] = o + leftSize;
        if (z.pw < 0) {
          rM[o + z.sz - 1] = INT_MAX;
          nl += z.sz;
        }
      }
      const int nbk = (nl + 63) >> 6;
      for (int b = 0; b < nbk; b++) {
        const int bs = b << 6, be = min(nl, bs + 64);
        uint64_t cur = 0;
        for (int i = bs; i < be; i++) {
          const int w = rM[i];
          while (cur) {
            const int t = 63 - __builtin_clzll(cur);
            if (rM[bs + t] >= w) cur ^= uint64_t(1) << t;
            else break;
          }
          cur |= uint64_t(1) << (i - bs);
          rmask[i] = cur;
        }
      }
      rbm.resize(size_t(nbk));
      for (int b = 0; b < nbk; b++) {
        const int be = min(nl, (b << 6) + 64) - 1;
        rbm[size_t(b)] = rM[(b << 6) + __builtin_ctzll(rmask[be])];
      }
      rtb.resize(size_t(spSize(nbk)));
      spBuild<int, true>(rbm.data(), nbk, rtb.data());
      rnb = nbk;
    };
    auto rmq_in = [&](int l, int r) {
      const int bs = l & ~63;
      return rM[bs + __builtin_ctzll(rmask[r] & (~uint64_t(0) << (l - bs)))];
    };
    auto rmq_bottleneck = [&](int a, int b) {
      int l = rpos[a], r = rpos[b];
      if (l > r) swap(l, r);
      r--;
      const int bl = l >> 6, bh = r >> 6;
      if (bl == bh) return rmq_in(l, r);
      int res = min(rmq_in(l, (bl << 6) + 63), rmq_in(bh << 6, r));
      if (bh - bl > 1) {
        res = min(res, rbm[size_t(SpTable<int, true, uint32_t>(rbm.data(), rnb, rtb.data()).query(bl + 1, bh - 1))]);
      }
      return res;
    };

    bool useRmq = false;
    int phaseNc = 0;
    int64_t budget = 0;
    auto phase_begin = [&](int nc, int64_t sizeVmu, int64_t nExcluded) {
      wsteps = 0;
      budget = 4 * (sizeVmu + nExcluded);
      phaseNc = nc;
      useRmq = false;
    };
    auto bottleneck = [&](int a, int b) -> int {
      if (useRmq) return rmq_bottleneck(a, b);
      const int w = bottleneck_walk(a, b);
      if (wsteps > budget) {
        rmq_build(phaseNc);
        useRmq = true;
      }
      return w;
    };

    auto gallop = [&](int x, int wakeBound) -> int {
      const int* tauX = nsr + vs[x].off;
      int good = mu - 2, d = 1, bad = -1;
      while (true) {
        const int p = good - d;
        if (p < 0) break;
        if (tauX[p] > wakeBound) {
          bad = p;
          break;
        }
        good = p;
        d <<= 1;
      }
      int lo = bad + 1, hi = good;
      while (lo < hi) {
        const int md = (lo + hi) >> 1;
        if (tauX[md] <= wakeBound) hi = md;
        else lo = md + 1;
      }
      return lo;
    };
    auto schedule = [&](const CoreEdge& c, int wakeBound) {
      if (c.sim <= wakeBound) return;
      int cn = INT_MAX;
      if (vs[c.u].tau <= wakeBound) cn = gallop(c.u, wakeBound);
      if (vs[c.v].tau <= wakeBound) cn = min(cn, gallop(c.v, wakeBound));
      const int nu = cn + 1;
      bucket[nu].push_back(int(c.ko));
    };

    for (; mu >= 2; mu--) {
      const int sizeVmu = sizeV[mu];
      for (int x = 0; x < sizeVmu; x++) vs[x].tau = nsr[vs[x].off + uint32_t(mu - 2)];
      fresh.clear();
      excluded.clear();
      int64_t ns = 0;
      {
        vector<int>& bk = bucket[mu];
        const size_t nWoken = bk.size();
        constexpr size_t P1 = 24, P2 = 12;
        auto st2 = [&](size_t q) {
          if (q >= nWoken) return;
          const SimEdge& e = rec[bk[q]];
          _mm_prefetch((const char*)&vs[e.u], _MM_HINT_T0);
          _mm_prefetch((const char*)&vs[e.v], _MM_HINT_T0);
        };
        for (size_t q = 0; q < min(nWoken, P1); q++) _mm_prefetch((const char*)&rec[bk[q]], _MM_HINT_T0);
        for (size_t q = 0; q < min(nWoken, P2); q++) st2(q);
        for (size_t q = 0; q < nWoken; q++) {
          if (q + P1 < nWoken) _mm_prefetch((const char*)&rec[bk[q + P1]], _MM_HINT_T0);
          st2(q + P2);
          const int k = bk[q];
          const SimEdge& e = rec[k];
          const CoreEdge c{min(e.sim, min(vs[e.u].tau, vs[e.v].tau)), e.sim, e.u, e.v, uint32_t(k)};
          woken.push_back(c);
        }
        vector<int>().swap(bk);
      }

      {
        constexpr int PD = 16;
        for (int64_t q = 0; q < nPrev; q++) {
          if (q + PD < nPrev) {
            _mm_prefetch((const char*)&vs[Fprev[q + PD].u], _MM_HINT_T0);
            _mm_prefetch((const char*)&vs[Fprev[q + PD].v], _MM_HINT_T0);
          }
          CoreEdge f = Fprev[q];
          const int wMu = min(f.sim, min(vs[f.u].tau, vs[f.v].tau));
          if (wMu == f.w) Fprev[ns++] = f;
          else {
            f.w = wMu;
            fresh.push_back(f);
          }
        }
      }

      fresh.insert(fresh.end(), woken.begin(), woken.end());
      woken.clear();
      {
        const int64_t k0 = ooff[sizeV[mu + 1]], k1 = ooff[sizeVmu];
        constexpr int64_t PD = 16;
        for (int64_t k = k0; k < k1; k++) {
          if (k + PD < k1) _mm_prefetch((const char*)&vs[rec[k + PD].v], _MM_HINT_T0);
          const SimEdge& e = rec[k];
          fresh.push_back({min(e.sim, min(vs[e.u].tau, vs[e.v].tau)), e.sim, e.u, e.v, uint32_t(k)});
        }
      }
      sort_desc_by_rank(fresh, ftmp, K, [](const CoreEdge& c) { return c.w; }, cnt);

      int nc = 0, touched = 0;
      int64_t nout = 0;
      sc.clear();
      auto find_root = [&](int x, int w) -> int {
        VertexState& s = vs[x];
        if (s.stamp != mu) {
          s.stamp = mu;
          s.p = x;
          s.up = x;
          s.tr = ~x;
          s.fe = int(nout);
          touched++;
          if (w < s.tau) sc.push_back({s.tau, x, w});
          if (mu == 2) e1[x] = w;
          return x;
        }
        while (vs[x].p != x) {
          int g = vs[vs[x].p].p;
          vs[x].p = g;
          x = g;
        }
        return x;
      };
      auto join = [&](const CoreEdge& c) -> bool {
        const int a = find_root(c.u, c.w), b = find_root(c.v, c.w);
        if (a == b) return false;
        const int j = nc++;
        const int ra = vs[a].tr, rb = vs[b].tr;
        const int sa = ra >= 0 ? kn[ra].sz : 1, sb = rb >= 0 ? kn[rb].sz : 1;
        kn[j] = {c.w, -1, ra, rb, sa + sb, 0};
        if (sa < sb) {
          VertexState& va = vs[a];
          va.p = b;
          va.up = b;
          va.tr = j;
          va.uw = c.w;
          vs[b].tr = j;
        } else {
          VertexState& vb = vs[b];
          vb.p = a;
          vb.up = a;
          vb.tr = j;
          vb.uw = c.w;
          vs[a].tr = j;
        }
        Fmu[nout++] = c;
        return true;
      };
      auto proc = [&](const CoreEdge& c) {
        if (!join(c)) excluded.push_back(c);
      };
      {
        const size_t ne = fresh.size();
        size_t i = 0;
        int64_t j = 0;
        constexpr int PD = 12;
        const VertexState* vsp = vs.data();
        auto pf_fresh = [&](size_t q) {
          if (q < ne) {
            _mm_prefetch((const char*)(vsp + fresh[q].u), _MM_HINT_T0);
            _mm_prefetch((const char*)(vsp + fresh[q].v), _MM_HINT_T0);
          }
        };
        auto pf_forest = [&](int64_t q) {
          if (q < ns) {
            _mm_prefetch((const char*)(vsp + Fprev[q].u), _MM_HINT_T0);
            _mm_prefetch((const char*)(vsp + Fprev[q].v), _MM_HINT_T0);
          }
        };
        for (int q = 0; q < PD; q++) {
          pf_fresh(q);
          pf_forest(q);
        }
        while (true) {
          if (j < ns) {
            if (i < ne && fresh[i].w > Fprev[j].w) {
              pf_fresh(i + PD);
              proc(fresh[i++]);
            } else {
              pf_forest(j + PD);
              proc(Fprev[j++]);
            }
          } else if (i < ne) {
            pf_fresh(i + PD);
            proc(fresh[i++]);
          } else {
            break;
          }
        }
      }

      Level lev;
      lev.lv.resize(touched);
      {
        int base = 0;
        for (int q = nc - 1; q >= 0; q--) {
          KrtNode& z = kn[q];
          if (z.pw < 0) {
            z.off = base;
            base += z.sz;
          }
          const int o = z.off;
          int c = z.lc, leftSize;
          if (c >= 0) {
            kn[c].off = o;
            kn[c].pw = z.w;
            leftSize = kn[c].sz;
          } else {
            lev.lv[o] = ord[~c];
            leftSize = 1;
          }
          c = z.rc;
          if (c >= 0) {
            kn[c].off = o + leftSize;
            kn[c].pw = z.w;
          } else lev.lv[o + leftSize] = ord[~c];
        }
      }

      {
        int kept = 0;
        for (int q = 0; q < nc; q++) kept += kn[q].w != kn[q].pw;
        lev.kw.resize(kept);
        lev.kpw.resize(kept);
        lev.klo.resize(kept);
        lev.khi.resize(kept);
        for (int q = 0, r = 0; q < nc; q++) {
          const KrtNode& z = kn[q];
          if (z.w == z.pw) continue;
          lev.kw[r] = z.w;
          lev.kpw[r] = z.pw;
          lev.klo[r] = z.off;
          lev.khi[r] = z.off + z.sz - 1;
          r++;
        }
      }

      for (int x = 0; x < sizeVmu; x++)
        if (vs[x].stamp != mu) sc.push_back({vs[x].tau, x, -1});
      sort_desc_by_rank(sc, sctmp, K, [](const SingletonRec& s) { return s.tau; }, cnt);
      lev.ss.resize(sc.size());
      lev.se.resize(sc.size());
      lev.sv.resize(sc.size());
      for (size_t q = 0; q < sc.size(); q++) {
        lev.ss[q] = sc[q].tau;
        lev.se[q] = sc[q].e;
        lev.sv[q] = ord[sc[q].x];
      }
      sink(mu, lev);

      {
        size_t nq = 0;
        for (const CoreEdge& c : excluded)
          if (c.w != c.sim) excluded[nq++] = c;
        excluded.resize(nq);
      }
      if (!excluded.empty()) {
        phase_begin(nc, sizeVmu, int64_t(excluded.size()));
        const size_t nExcluded = excluded.size();
        constexpr size_t P1 = 24, P2 = 12, P3 = 5;
        auto st1 = [&](size_t q) {
          if (q < nExcluded) {
            _mm_prefetch((const char*)&vs[excluded[q].u], _MM_HINT_T0);
            _mm_prefetch((const char*)&vs[excluded[q].v], _MM_HINT_T0);
          }
        };
        auto st2 = [&](size_t q) {
          if (q >= nExcluded) return;
          const VertexState &a = vs[excluded[q].u], &b = vs[excluded[q].v];
          _mm_prefetch((const char*)&Fmu[a.fe], _MM_HINT_T0);
          _mm_prefetch((const char*)&Fmu[b.fe], _MM_HINT_T0);
          _mm_prefetch((const char*)(nsr + a.off + uint32_t(mu - 3)), _MM_HINT_T0);
          _mm_prefetch((const char*)(nsr + b.off + uint32_t(mu - 3)), _MM_HINT_T0);
        };
        auto st3 = [&](size_t q) {
          if (q >= nExcluded) return;
          const CoreEdge& c = excluded[q];
          const CoreEdge& f = Fmu[vs[c.u].fe];
          const CoreEdge& g = Fmu[vs[c.v].fe];
          _mm_prefetch((const char*)&vs[f.u == c.u ? f.v : f.u], _MM_HINT_T0);
          _mm_prefetch((const char*)&vs[g.u == c.v ? g.v : g.u], _MM_HINT_T0);
        };
        for (size_t q = 0; q < min(nExcluded, P1); q++) st1(q);
        for (size_t q = 0; q < min(nExcluded, P2); q++) st2(q);
        for (size_t q = 0; q < min(nExcluded, P3); q++) st3(q);
        for (size_t q = 0; q < nExcluded; q++) {
          st1(q + P1);
          st2(q + P2);
          st3(q + P3);
          const CoreEdge& c = excluded[q];
          const CoreEdge& f = Fmu[vs[c.u].fe];
          const CoreEdge& g = Fmu[vs[c.v].fe];
          const int uNbr = f.u == c.u ? f.v : f.u, vNbr = g.u == c.v ? g.v : g.u;
          const int inner = uNbr == vNbr ? vs[uNbr].tau : bottleneck(uNbr, vNbr);
          const int endpointCert = min(min(f.sim, g.sim), inner);
          schedule(c, max(endpointCert, c.w));
        }
      }

      swap(Fprev, Fmu);
      nPrev = nout;
    }
  }

  {
    Level l1;
    l1.se.resize(n);
    l1.sv.resize(n);
    for (int x = 0; x < n; x++) {
      l1.sv[x] = ord[x];
      l1.se[x] = e1[x];
    }
    sink(1, l1);
  }
}
