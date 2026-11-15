#pragma once
#include "io.hpp"

struct BuildTimes {
  double raw;
  double ser;
  uint64_t total;
};

static bool fast_route_fits(const Graph& G) {
  const uint64_t n = uint64_t(G.n), m = uint64_t(G.E.size());
  const uint64_t fastBytes = n * (8 * 16) + m * 2 * 48;
  return fastBytes <= (uint64_t(32) << 30);
}

static BuildTimes build_index_file(Graph& G, Engine& eng, const string& path) {
  BuildTimes bt;
  const auto t0 = Clock::now();
  const bool fast = fast_route_fits(G);
  G.finalize(fast);
  trim_heap();

  vector<int> baseLeafOrder;
  baseLeafOrder.swap(G.baseLeafOrder);
  eng.CC = new_compact_core(G);
  CoreBase* cc = eng.CC.get();
  cc->init_ids(G, baseLeafOrder);
  vector<int>().swap(baseLeafOrder);
  trim_heap();
  krtLeaves.assign(size_t(G.Delta) + 3, {});
  krtGapW.assign(size_t(G.Delta) + 3, {});

  const int* nid = cc->nidv.data();
  const int* oid = cc->oidv.data();
  eng.D.buildL(G, oid);
  trim_heap();
  cc->prepare_lcodes(G, eng.D);

  BlockOut o = begin_index(path);
  save_index_prefix(o, G, eng.D, *cc);
  eng.D.release_l_written();
  trim_heap();

  const auto sink = [cc, &o](int mu, Level& l) {
    if (mu >= 3) keep_krt_gaps(mu, l);
    cc->encode_level(mu, l);
    l.free();
    cc->finish_level(mu, o);
  };
  build_levels(G, sink);
  cc->release_lcodes();
  eng.D.release_l_rest();
  cc->free_scratch();
  trim_heap();

  const uint64_t ctailOff = o.total;
  cc->save_tail(o);
  const uint64_t ctailLen = o.total - ctailOff;
  eng.D.buildH(G, nid);
  vector<int>().swap(G.deg);
  trim_heap();

  vector<int64_t> sizeBH;
  build_bh(G, eng.D, *cc, nid, oid, o, sizeBH, fast);
  vector<vector<int>>().swap(krtLeaves);
  vector<vector<int>>().swap(krtGapW);
  vector<int>().swap(cc->nidv);
  vector<int>().swap(cc->oidv);
  trim_heap();
  save_index_end(o, sizeBH, ctailOff, ctailLen);
  finish_index(o);

  bt.raw = secs(t0, Clock::now());
  bt.ser = o.sum;
  bt.total = o.total;
  return bt;
}
