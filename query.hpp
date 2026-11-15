#pragma once
#include "hborder.hpp"
#include "lborder.hpp"

struct Engine {
  const Graph& G;
  Borders D;
  unique_ptr<CoreBase> CC;
  vector<int> label;
  vector<pair<int, int>> out, rng;
  vector<int> lcore;
  size_t ncore;
  vector<uint64_t> seenBits;
  vector<pair<int, int>> borderPairs;
  explicit Engine(const Graph& g) : G(g) {}
  void prepare() {
    label.assign(G.n, -1);
    seenBits.assign(G.n / 64 + 1, 0);
    borderPairs.assign(4096, {0, 0});
  }

  bool no_core(int epsRank, int mu) const {
    return int64_t(epsRank) >= int64_t(CC->amax[mu]);
  }

  void query_h_borders(int epsRank, int mu, size_t& nbuf) {
    const Borders::HLists& B = D.hLists;
    const int64_t base = B.off[mu], sizeBHmu = B.off[mu + 1] - base;
    const int* WB = B.WB.data() + base;
    const int* LO = B.LO.data() + base;
    const int* X = B.X.data() + base;
    const uint32_t* U = B.U.data() + base;
    const int hb = B.hb;
    int64_t sizeBHmuEps = 0;
    {
      int k = 0;
      for (;; k++) {
        const int64_t q = (int64_t(1) << k) - 1;
        if (q >= sizeBHmu || WB[q] < epsRank) break;
      }
      if (k > 0) {
        int64_t a = int64_t(1) << (k - 1), b = min((int64_t(1) << k) - 1, sizeBHmu);
        while (a < b) {
          const int64_t md = (a + b) >> 1;
          if (WB[md] >= epsRank) a = md + 1;
          else b = md;
        }
        sizeBHmuEps = a;
      }
    }

    auto report = [&](int64_t i) {
      const uint32_t u = U[i];
      if (nbuf == borderPairs.size()) borderPairs.resize(2 * nbuf + 1024);
      borderPairs[nbuf++] = {X[i], label[u]};
    };
    const int64_t nbl = (sizeBHmu + (int64_t(1) << hb) - 1) >> hb;
    const int64_t nfull = sizeBHmuEps >> hb;
    if (nfull > 0) {
      const int* BM = B.BM.data() + B.boff[mu];
      const uint32_t* ST = B.ST.data() + B.soff[mu];
      const SpTable<int, true, uint32_t> rmq(BM, int(nbl), ST);
      int64_t sa[130], sb[130];
      int sp = 0;
      sa[sp] = 0;
      sb[sp] = nfull - 1;
      sp++;
      while (sp > 0) {
        sp--;
        const int64_t a = sa[sp], b = sb[sp];
        const int64_t i = rmq.query(int(a), int(b));
        if (BM[i] >= epsRank) continue;
        const int64_t e = (i + 1) << hb;
        for (int64_t k = i << hb; k < e; k++)
          if (LO[k] < epsRank) report(k);
        const bool lft = a <= i - 1, rgt = i + 1 <= b;
        if (lft && rgt && (i - a) < (b - i)) {
          sa[sp] = i + 1;
          sb[sp] = b;
          sp++;
          sa[sp] = a;
          sb[sp] = i - 1;
          sp++;
        } else {
          if (lft) {
            sa[sp] = a;
            sb[sp] = i - 1;
            sp++;
          }
          if (rgt) {
            sa[sp] = i + 1;
            sb[sp] = b;
            sp++;
          }
        }
      }
    }
    for (int64_t k = nfull << hb; k < sizeBHmuEps; k++)
      if (LO[k] < epsRank) report(k);
  }

  void query_rank(int epsRank, int mu) {
    out.clear();
    ncore = 0;
    if (mu > G.Delta + 1) return;
    if (no_core(epsRank, mu)) return;
    const float epsProxy = G.rank_proxy(epsRank);
    const size_t nEpsMu = CC->core_phase(epsRank, mu, label.data(), out, lcore, int(sim_key8(epsProxy)));
    ncore = nEpsMu;
    size_t nbuf = 0;
    auto room = [&](size_t m) {
      if (nbuf + m > borderPairs.size()) borderPairs.resize(2 * (nbuf + m) + 1024);
    };

    query_h_borders(epsRank, mu, nbuf);
    {
      const int kq = mu - 2;
      const uint16_t epsKey = Borders::key16(epsProxy);
      int curLabel = INT_MIN;
      bool ddup = true;
      size_t runStart = nbuf;
      auto end_run = [&]() {
        if (ddup)
          for (size_t i = runStart; i < nbuf; i++) {
            const int x = borderPairs[i].first;
            seenBits[x >> 6] &= ~(uint64_t(1) << (x & 63));
          }
        runStart = nbuf;
      };
      auto put_border = [&](int x, int labelU) {
        if (ddup) {
          uint64_t& w = seenBits[x >> 6];
          const uint64_t b = uint64_t(1) << (x & 63);
          if (w & b) return;
          w |= b;
        }
        borderPairs[nbuf++] = {x, labelU};
      };
      const Borders::LH* lh = D.lh.data();
      const int* lvtx = D.lvtx.data();
      const float* lsim = D.lsim.data();
      const uint16_t* ld = D.ld.data();
      const int lbb = D.lbb, lsb = D.lsb, lshort = D.lshort, lcap = D.lcap;
      const __m256 epsVec = _mm256_set1_ps(epsProxy);
      const int64_t lsimN = int64_t(D.lsim.size());
      auto pmask = [&](int64_t b0, uint32_t mk, int labelU) {
        room(32);
        while (mk) {
          const int i = __builtin_ctz(mk);
          mk &= mk - 1;
          put_border(lvtx[b0 + i], labelU);
        }
      };
      auto lblock = [&](int64_t b0, int labelU) { pmask(b0, mask_ge_f32(lsim + b0, epsVec), labelU); };

      const size_t nlcore = lcore.size();
      const int* lcp = lcore.data();
      for (size_t i = 0; i < nlcore; i++) {
        const int u = lcp[i];
        if (i + 8 < nlcore) {
          _mm_prefetch((const char*)&lh[lcp[i + 8]], _MM_HINT_T0);
          _mm_prefetch((const char*)&label[lcp[i + 8]], _MM_HINT_T0);
        }
        if (i + 2 < nlcore) {
          const uint32_t ahead = lh[lcp[i + 2]].off;
          _mm_prefetch((const char*)(lsim + ahead), _MM_HINT_T0);
          _mm_prefetch((const char*)(lvtx + ahead), _MM_HINT_T0);
          _mm_prefetch((const char*)(ld + ahead), _MM_HINT_T0);
        }

        const Borders::LH lhu = lh[u];
        if (lhu.dmin16 > kq || epsKey > lhu.smax16) continue;
        const int labelU = label[u];
        if (labelU != curLabel) {
          curLabel = labelU;
          end_run();
          ddup = false;
          for (size_t j = i + 1; j < nlcore; j++) {
            const int w = lcp[j];
            if (label[w] != labelU) break;
            const Borders::LH lhw = lh[w];
            if (!(lhw.dmin16 > kq || epsKey > lhw.smax16)) {
              ddup = true;
              break;
            }
          }
        }

        const int64_t off = lhu.off, len = int64_t(lh[u + 1].off) - off;
        if (len <= lshort && off + 32 <= lsimN) {
          int64_t j = off;
          for (const int64_t e = off + len; j < e; j++) {
            int dx = ld[j];
            if (dx == lcap) dx = G.deg[lvtx[j]];
            if (dx > kq) break;
          }
          if (j > off)
            pmask(off, mask_ge_f32(lsim + off, epsVec) & uint32_t((uint64_t(1) << (j - off)) - 1), labelU);
          continue;
        }
        if (len <= lshort) {
          room(size_t(len));
          for (int64_t j = off, e = off + len; j < e; j++) {
            int dx = ld[j];
            if (dx == lcap) dx = G.deg[lvtx[j]];
            if (dx > kq) break;
            if (lsim[j] >= epsProxy) put_border(lvtx[j], labelU);
          }
          continue;
        }

        const Borders::LLong& llu = D.llong[D.longId(u)];
        const int64_t cnt = kq >= llu.dmax ? len
                                           : (llu.wide ? int64_t(D.lpt32[llu.pt + (kq - llu.dmin)])
                                                       : int64_t(D.lpt16[llu.pt + (kq - llu.dmin)]));
        const int64_t full = cnt >> lbb;
        if (off + (full << lbb) + 32 <= lsimN) {
          const int64_t j0 = off + (full << lbb), nt = cnt - (full << lbb);
          if (nt > 0) pmask(j0, mask_ge_f32(lsim + j0, epsVec) & uint32_t((uint64_t(1) << nt) - 1), labelU);
        } else {
          room(size_t(cnt - (full << lbb)));
          for (int64_t j = off + (full << lbb), e = off + cnt; j < e; j++)
            if (lsim[j] >= epsProxy) put_border(lvtx[j], labelU);
        }
        const float* bk = D.lbk.data() + llu.bo;
        const int64_t fs = full >> lsb;
        {
          uint32_t mk = mask_ge_f32(bk + (fs << lsb), epsVec) & uint32_t((uint64_t(1) << (full - (fs << lsb))) - 1);
          while (mk) {
            const int b = __builtin_ctz(mk);
            mk &= mk - 1;
            lblock(off + ((fs << lsb) + b) * (int64_t(1) << lbb), labelU);
          }
        }
        if (fs > 0) {
          const float* sk = D.lsk.data() + llu.so;
          const int* sp = D.lsp.data() + llu.sp;
          const SpTable<float, false, int> rmq(sk, llu.nsb, sp);
          rng.push_back({0, int(fs) - 1});
          while (!rng.empty()) {
            const auto pr = rng.back();
            rng.pop_back();
            const int l = pr.first, r = pr.second;
            const int k = rmq.query(l, r);
            if (sk[k] < epsProxy) continue;
            const int64_t k0 = int64_t(k) << lsb;
            uint32_t mk = mask_ge_f32(bk + k0, epsVec);
            while (mk) {
              const int b = __builtin_ctz(mk);
              mk &= mk - 1;
              lblock(off + (k0 + b) * (int64_t(1) << lbb), labelU);
            }
            if (l < k) rng.push_back({l, k - 1});
            if (k < r) rng.push_back({k + 1, r});
          }
        }
      }
      end_run();
    }

    out.insert(out.end(), borderPairs.begin(), borderPairs.begin() + nbuf);
  }

  void query(const Threshold& eps, int mu) { query_rank(G.thr_rank(eps), mu); }
};
