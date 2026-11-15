#pragma once
#include "borders.hpp"

inline void Borders::buildL(const Graph& G, const int* oid) {
  const int n = G.n;
  const int* deg = G.deg.data();
  const int64_t* off = G.off.data();
  const int* nbr = G.nbr.data();
  hvec<int> sizeL(n), dmn(n), dmx(n);
  for (int u = 0; u < n; u++) {
    const int origU = oid[u], du = deg[origU];
    int len = 0, mn = INT_MAX, mx = 0;
    for (int64_t i = off[origU], e = off[origU + 1]; i < e; i++) {
      if (i + 16 < e) _mm_prefetch((const char*)&deg[nbr[i + 16]], _MM_HINT_T0);
      const int x = nbr[i], dx = deg[x];
      if (dx < du) {
        len++;
        mn = min(mn, dx);
        mx = max(mx, dx);
      }
    }
    sizeL[u] = len;
    dmn[u] = mn;
    dmx[u] = mx;
  }

  lh.resize(n + 1);
  llb.assign(n / 64 + 1, 0);
  llc.assign(n / 64 + 1, 0);
  int64_t s = 0, spt16 = 0, spt32 = 0, sbo = 0, sso = 0, ssp = 0;
  for (int u = 0; u < n; u++) {
    if ((u & 63) == 0) llc[u >> 6] = uint32_t(llong.size());
    const int len = sizeL[u];
    lh[u] = {uint32_t(s), uint16_t(min(dmn[u], lcap)), 0};
    s += len;
    if (len > lshort) {
      const int nb = len >> lbb, nsb = nb >> lsb, wide = len > 65535;
      llb[u >> 6] |= uint64_t(1) << (u & 63);
      llong.push_back({wide ? spt32 : spt16, sbo, sso, ssp, dmn[u], dmx[u], nb, nsb, wide});
      (wide ? spt32 : spt16) += dmx[u] - dmn[u];
      sbo += nb;
      sso += nsb;
      ssp += spSize(nsb);
    }
  }

  lh[n] = {uint32_t(s), lcap, 0};
  lvtx.resize(s);
  lsim.resize(s);
  ld.resize(s);
  lpt16.resize(spt16);
  lpt32.resize(spt32);
  lbk.assign(sbo + 32, -INFINITY);
  lsk.resize(sso);
  lsp.resize(ssp);

  vector<int> loc(G.Delta + 2);
  for (int u = 0; u < n; u++) {
    const int len = sizeL[u];
    if (!len) continue;
    const int origU = oid[u];
    const int du = deg[origU], mn = dmn[u], mx = dmx[u], R = mx - mn + 1;
    for (int k = 0; k < R; k++) loc[k] = 0;
    for (int64_t i = off[origU], e = off[origU + 1]; i < e; i++) {
      if (i + 16 < e) _mm_prefetch((const char*)&deg[nbr[i + 16]], _MM_HINT_T0);
      const int x = nbr[i], dx = deg[x];
      if (dx < du) loc[dx - mn]++;
    }

    const LLong* llu = len > lshort ? &llong[longId(u)] : nullptr;
    int acc = 0;
    for (int k = 0; k < R; k++) {
      const int y = loc[k];
      loc[k] = acc;
      acc += y;
      if (llu && k + 1 < R) {
        if (llu->wide) lpt32[llu->pt + k] = uint32_t(acc);
        else lpt16[llu->pt + k] = uint16_t(acc);
      }
    }

    const int64_t base = lh[u].off;
    float smax = -INFINITY;
    for (int64_t i = off[origU], e = off[origU + 1]; i < e; i++) {
      const int x = nbr[i], dx = deg[x];
      if (dx >= du) continue;
      const int64_t p = base + loc[dx - mn]++;
      lvtx[p] = x;
      lsim[p] = G.vals[G.nsr[i]];
      ld[p] = uint16_t(min(dx, lcap));
      smax = max(smax, G.vals[G.nsr[i]]);
    }
    lh[u].smax16 = key16(smax);

    if (llu) {
      float* bk = lbk.data() + llu->bo;
      for (int k = 0; k < llu->nb; k++) {
        const float* y = lsim.data() + base + (int64_t(k) << lbb);
        float bmax = y[0];
        for (int i = 1; i < (1 << lbb); i++) bmax = max(bmax, y[i]);
        bk[k] = bmax;
      }
      float* sk = lsk.data() + llu->so;
      for (int k = 0; k < llu->nsb; k++) {
        float sbmax = bk[k << lsb];
        for (int i = 1; i < (1 << lsb); i++) sbmax = max(sbmax, bk[(k << lsb) + i]);
        sk[k] = sbmax;
      }
      spBuild<float, false>(sk, llu->nsb, lsp.data() + llu->sp);
    }
  }
}

template <class R>
void CompactCore<R>::prepare_lcodes(const Graph& G, const Borders& D) {
  lMinMu.assign(size_t(n), INT_MAX);
  for (int u = 0; u < n; u++) {
    int dm = INT_MAX;
    for (uint32_t p = D.lh[u].off, e = D.lh[u + 1].off; p < e; p++) {
      int dx = D.ld[p];
      if (dx == D.lcap) dx = G.deg[D.lvtx[p]];
      dm = min(dm, dx);
    }
    lMinMu[u] = dm == INT_MAX ? INT_MAX : dm + 2;
  }
  lBorders = &D;
  lGraph = &G;
  lKeyMax.assign(size_t(D.lh[n].off), 0);
  lKeyEnd.assign(size_t(n), 0);
  for (int u = 0; u < n; u++) {
    const uint32_t b = D.lh[u].off, e = D.lh[u + 1].off;
    uint8_t m = 0;
    for (uint32_t p = b; p < e; p++) {
      m = max(m, sim_key8(D.lsim[p]));
      lKeyMax[p] = m;
    }
    lKeyEnd[u] = e;
  }
}

template <class R>
uint8_t CompactCore<R>::lkey(uint32_t u, int mu) {
  if (lMinMu[u] > mu) return 0;
  const Borders& D = *lBorders;
  const Graph& G = *lGraph;
  auto dg = [&](uint32_t p) {
    int dx = D.ld[p];
    if (dx == D.lcap) dx = G.deg[D.lvtx[p]];
    return dx;
  };
  uint32_t cntEnd = lKeyEnd[u];
  while (dg(cntEnd - 1) > mu - 2) cntEnd--;
  lKeyEnd[u] = cntEnd;
  return lKeyMax[cntEnd - 1];
}
