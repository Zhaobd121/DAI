#pragma once
#include "core.hpp"

static inline uint32_t mask_ge_f32(const float* x, __m256 tv) {
  uint32_t m = 0;
  for (int k = 0; k < 4; k++)
    m |= uint32_t(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x + 8 * k), tv, _CMP_GE_OQ))) << (8 * k);
  return m;
}

struct Borders {
  static constexpr int lbb = 5, lsb = 5, lshort = 32, lcap = 65535;
  int hi = 2;

  int vBits = 1;
  vector<int64_t> candOff, candRecOff;
  vector<uint8_t> candRecBytes, candRec;
  uint64_t cand_record(int mu, int64_t i) const {
    return load8(candRec.data() + candRecOff[mu] + i * int64_t(candRecBytes[mu]));
  }
  uint32_t cand_vertex(int mu, int64_t i) const {
    return uint32_t(cand_record(mu, i) & ((uint64_t(1) << vBits) - 1));
  }
  template <class FV>
  void pack_cand_level(int mu, int64_t len, FV vv) {
    const int R = candRecBytes[mu];
    uint8_t* q = candRec.data() + candRecOff[mu];
    for (int64_t i = 0; i < len; i++) {
      const uint64_t x = uint64_t(uint32_t(vv(i)));
      memcpy(q + i * R, &x, 8);
    }
  }
  void layout_cands() {
    const int NL = hi + 3;
    candRecOff.assign(NL, 0);
    for (int mu = 0; mu + 1 < NL; mu++) {
      const int64_t len = candOff[mu + 1] - candOff[mu];
      candRecOff[mu + 1] = candRecOff[mu] + len * candRecBytes[mu] + 8;
    }
    candRec.assign(candRecOff[NL - 1] + 8, 0);
  }

  struct KeptEntry { uint32_t u; int wb, gi; };

  struct HLists {
    static constexpr int hb = 5;
    int muMax = 1;
    vector<int64_t> off, boff, soff;
    vector<int> WB, LO, X;
    vector<uint32_t> U;
    vector<int> BM;
    vector<uint32_t> ST;
  } hLists;

  struct LH { uint32_t off; uint16_t dmin16, smax16; };
  struct LLong { int64_t pt, bo, so, sp; int dmin, dmax, nb, nsb, wide; };
  static uint16_t key16(float x) { return uint16_t(Graph::fkey(x) >> 16); }
  vector<LH> lh;
  vector<uint64_t> llb;
  vector<uint32_t> llc;
  vector<int> lvtx, lsp;
  vector<float> lsim, lbk, lsk;
  vector<uint16_t> ld, lpt16;
  vector<uint32_t> lpt32;
  vector<LLong> llong;

  void release_l_written() {
    vector<LH>().swap(lh);
    vector<uint64_t>().swap(llb);
    vector<uint32_t>().swap(llc);
    vector<LLong>().swap(llong);
    vector<float>().swap(lsim);
    vector<uint16_t>().swap(lpt16);
    vector<uint32_t>().swap(lpt32);
    vector<float>().swap(lbk);
    vector<float>().swap(lsk);
    vector<int>().swap(lsp);
  }
  void release_l_rest() {
    vector<int>().swap(lvtx);
    vector<uint16_t>().swap(ld);
  }

  int longId(int u) const {
    const int w = u >> 6;
    return int(llc[w]) + __builtin_popcountll(llb[w] & ((uint64_t(1) << (u & 63)) - 1));
  }

  void buildH(const Graph& G, const int* nid);
  void buildL(const Graph& G, const int* oid);
};
