#pragma once
#include "query.hpp"

struct BlockIn {
  FILE* f = nullptr;
  string path;
  void get(void* p, size_t len) {
    if (len && fread(p, 1, len, f) != len) die("cannot read " + path + " (short read)");
  }
  uint64_t next() {
    char h[24];
    get(h, 24);
    uint64_t len;
    memcpy(&len, h + 16, 8);
    return len;
  }
  template <class T, class A>
  void vec(vector<T, A>& v) {
    const uint64_t len = next();
    v.resize(len / sizeof(T));
    get(v.data(), len);
  }
  uint64_t tell() const { return uint64_t(ftello(f)); }
  void seek(uint64_t pos) {
    if (fseeko(f, off_t(pos), SEEK_SET)) die("cannot seek " + path);
  }
};

struct IndexFooter {
  char magic[8];
  uint32_t version, complete;
  uint64_t ctailOff, ctailLen;
  uint64_t bdirOff, bdirLen;
  uint64_t fileLen;
};

static constexpr int kCoreHead = 16;

template <class R>
void CompactCore<R>::finish_level(int mu, BlockOut& o) {
  Lvl& c = levels[mu];
  uint32_t m = 0;
  if (!c.ss.empty()) m = max(m, uint32_t(c.ss[0]));
  if (!c.kw.empty()) m = max(m, uint32_t(c.kw[0]));
  amax[mu] = m;
  uint8_t k = 255;
  for (uint8_t b : c.lkl)
    if (b && b < k) k = b;
  for (uint8_t b : c.lks)
    if (b && b < k) k = b;
  kmin[mu] = k;
  const uint64_t w = uint64_t(c.sv.w) | uint64_t(c.ks.w) << 8 | uint64_t(c.kc.w) << 16 |
                     uint64_t(c.lql.w) << 24 | uint64_t(c.lqs.w) << 32;
  const uint64_t h[kCoreHead] = {uint64_t(mu), c.ss.size(), c.se.size(), c.kw.size(),
                                 c.kpw.size(), c.fd.size(), c.sv.b.size(), c.fst.size(),
                                 c.ks.b.size(), c.kc.b.size(), c.lql.b.size(), c.lqs.b.size(),
                                 c.lkl.size(), c.lks.size(), c.nlf, w};
  const pair<const void*, size_t> a[13] = {
      {c.ss.data(), c.ss.size() * sizeof(R)}, {c.se.data(), c.se.size() * sizeof(R)},
      {c.kw.data(), c.kw.size() * sizeof(R)}, {c.kpw.data(), c.kpw.size() * sizeof(R)},
      {c.fd.data(), c.fd.size() * sizeof(typename Lvl::FD)},
      {c.sv.b.data(), c.sv.b.size()}, {c.fst.data(), c.fst.size()}, {c.ks.b.data(), c.ks.b.size()},
      {c.kc.b.data(), c.kc.b.size()}, {c.lql.b.data(), c.lql.b.size()}, {c.lqs.b.data(), c.lqs.b.size()},
      {c.lkl.data(), c.lkl.size()}, {c.lks.data(), c.lks.size()}};
  uint64_t len = sizeof h;
  for (const auto& x : a) len += x.second;
  o.begin("c.lv", len);
  o.put(h, sizeof h);
  for (const auto& x : a) o.put(x.first, x.second);
  c = Lvl();
}

template <class R>
void CompactCore<R>::save_tail(BlockOut& o) const {
  o.vec("c.orig", orig.b);
  o.vec("c.amax", amax);
  o.vec("c.kmin", kmin);
  const vector<uint64_t> nl = {levels.size()};
  o.vec("c.nlv", nl);
}

template <class R>
void CompactCore<R>::load(BlockIn& in, const Graph& G, uint64_t end) {
  n = G.n;
  Delta = G.Delta;
  levels.assign(size_t(Delta) + 2, Lvl());
  while (in.tell() < end) {
    in.next();
    uint64_t h[kCoreHead];
    in.get(h, sizeof h);
    Lvl& c = levels[h[0]];
    c.ss.resize(h[1]);
    c.se.resize(h[2]);
    c.kw.resize(h[3]);
    c.kpw.resize(h[4]);
    c.fd.resize(h[5]);
    c.sv.b.resize(h[6]);
    c.fst.resize(h[7]);
    c.ks.b.resize(h[8]);
    c.kc.b.resize(h[9]);
    c.lql.b.resize(h[10]);
    c.lqs.b.resize(h[11]);
    c.lkl.resize(h[12]);
    c.lks.resize(h[13]);
    c.nlf = uint32_t(h[14]);
    c.sv.w = int(h[15] & 255);
    c.sv.mask = c.sv.w >= 8 ? ~0ull : ((uint64_t(1) << (8 * c.sv.w)) - 1);
    BitArr* b[4] = {&c.ks, &c.kc, &c.lql, &c.lqs};
    for (int k = 0; k < 4; k++) {
      b[k]->w = uint8_t(h[15] >> (8 * (k + 1)));
      b[k]->mask = (uint64_t(1) << b[k]->w) - 1;
    }
    in.get(c.ss.data(), c.ss.size() * sizeof(R));
    in.get(c.se.data(), c.se.size() * sizeof(R));
    in.get(c.kw.data(), c.kw.size() * sizeof(R));
    in.get(c.kpw.data(), c.kpw.size() * sizeof(R));
    in.get(c.fd.data(), c.fd.size() * sizeof(typename Lvl::FD));
    in.get(c.sv.b.data(), c.sv.b.size());
    in.get(c.fst.data(), c.fst.size());
    in.get(c.ks.b.data(), c.ks.b.size());
    in.get(c.kc.b.data(), c.kc.b.size());
    in.get(c.lql.b.data(), c.lql.b.size());
    in.get(c.lqs.b.data(), c.lqs.b.size());
    in.get(c.lkl.data(), c.lkl.size());
    in.get(c.lks.data(), c.lks.size());
  }
  orig.init(0, n - 1);
  in.vec(orig.b);
  in.vec(amax);
  in.vec(kmin);
}

static BlockOut begin_index(const string& path) {
  BlockOut o;
  o.path = path;
  o.fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (o.fd < 0) die("cannot open " + path);
  o.buf.resize(BlockOut::kBuf);
  const char magic[16] = {'D', 'A', 'I', 'I', 'N', 'D', 'E', 'X', 6, 0, 0, 0, 0, 0, 0, 0};
  o.put(magic, 16);
  return o;
}

static void save_index_prefix(BlockOut& o, const Graph& G, const Borders& D, const CoreBase& C) {
  const vector<int64_t> sc = {G.n, G.m, G.Delta, G.K, max(2, G.Delta + 1), G.Delta + 1, C.rbytes()};
  o.vec("scalars", sc);
  o.vec("g.vals", G.vals);
  o.vec("g.RP2", G.RP2);
  o.vec("g.RQ2", G.RQ2);
  bool cap = false;
  for (uint16_t d : D.ld)
    if (d == Borders::lcap) {
      cap = true;
      break;
    }
  if (cap) o.vec("g.deg", G.deg);
  else o.vec("g.deg", vector<int>());
  o.vec("l.lh", D.lh);
  o.vec("l.llb", D.llb);
  o.vec("l.llc", D.llc);
  o.begin("l.llong", D.llong.size() * sizeof(Borders::LLong));
  {
    BlockOut::Span s(o);
    for (const Borders::LLong& x : D.llong) {
      Borders::LLong r;
      memset((void*)&r, 0, sizeof r);
      r.pt = x.pt;
      r.bo = x.bo;
      r.so = x.so;
      r.sp = x.sp;
      r.dmin = x.dmin;
      r.dmax = x.dmax;
      r.nb = x.nb;
      r.nsb = x.nsb;
      r.wide = x.wide;
      o.raw(&r, sizeof r);
    }
  }
  o.vec("l.lvtx", D.lvtx);
  o.vec("l.lsim", D.lsim);
  o.vec("l.ld", D.ld);
  o.vec("l.lpt16", D.lpt16);
  o.vec("l.lpt32", D.lpt32);
  o.vec("l.lbk", D.lbk);
  o.vec("l.lsk", D.lsk);
  o.vec("l.lsp", D.lsp);
}

static void save_index_end(BlockOut& o, const vector<int64_t>& sizeBH, uint64_t ctailOff, uint64_t ctailLen) {
  const int muMax = int(sizeBH.size()) - 1;
  vector<int64_t> off(size_t(muMax) + 2, 0), boff(size_t(muMax) + 2, 0), soff(size_t(muMax) + 2, 0);
  for (int mu = 0; mu <= muMax; mu++) {
    const int64_t nbl = (sizeBH[mu] + (int64_t(1) << Borders::HLists::hb) - 1) >> Borders::HLists::hb;
    off[mu + 1] = off[mu] + sizeBH[mu];
    boff[mu + 1] = boff[mu] + nbl;
    soff[mu + 1] = soff[mu] + spSize(nbl);
  }
  const uint64_t bdirOff = o.total;
  o.vec("h.off", off);
  o.vec("h.boff", boff);
  o.vec("h.soff", soff);
  IndexFooter ft;
  memcpy(ft.magic, "DAIIXEND", 8);
  ft.version = 6;
  ft.complete = 1;
  ft.ctailOff = ctailOff;
  ft.ctailLen = ctailLen;
  ft.bdirOff = bdirOff;
  ft.bdirLen = o.total - bdirOff;
  ft.fileLen = o.total + 24 + sizeof ft;
  o.begin("footer", sizeof ft);
  o.put(&ft, sizeof ft);
}

static void finish_index(BlockOut& o) {
  {
    BlockOut::Span s(o);
    o.flush_buf();
    if (pwrite(o.fd, &o.nb, 4, 12) != 4 || fsync(o.fd) != 0 || close(o.fd) != 0) die("cannot write " + o.path);
  }
  o.fd = -1;
  vector<char>().swap(o.buf);
}

static void load_index(const string& path, Graph& G, Borders& D, unique_ptr<CoreBase>& C) {
  BlockIn in;
  in.path = path;
  in.f = fopen(path.c_str(), "rb");
  if (!in.f) die("cannot open " + path);
  setvbuf(in.f, nullptr, _IOFBF, 1 << 23);

  char magic[16];
  in.get(magic, 16);
  uint32_t ver;
  memcpy(&ver, magic + 8, 4);
  if (memcmp(magic, "DAIINDEX", 8)) die(path + " is not a DAI index");
  if (ver != 6) die("unknown index format version " + to_string(ver));

  IndexFooter ft;
  if (fseeko(in.f, -off_t(sizeof ft), SEEK_END)) die("cannot seek " + path);
  const uint64_t flen = in.tell() + sizeof ft;
  in.get(&ft, sizeof ft);
  if (memcmp(ft.magic, "DAIIXEND", 8) || ft.version != 6 || ft.complete != 1 || ft.fileLen != flen ||
      ft.ctailOff + ft.ctailLen > ft.bdirOff || ft.bdirOff + ft.bdirLen + 24 + sizeof ft != flen)
    die("index " + path + " is incomplete or damaged (footer)");

  in.seek(16);
  vector<int64_t> sc;
  in.vec(sc);
  G.n = int(sc[0]);
  G.Delta = int(sc[2]);
  G.K = int(sc[3]);
  D.hLists.muMax = int(sc[5]);
  in.vec(G.vals);
  in.vec(G.RP2);
  in.vec(G.RQ2);
  in.vec(G.deg);
  in.vec(D.lh);
  in.vec(D.llb);
  in.vec(D.llc);
  in.vec(D.llong);
  in.vec(D.lvtx);
  in.vec(D.lsim);
  in.vec(D.ld);
  in.vec(D.lpt16);
  in.vec(D.lpt32);
  in.vec(D.lbk);
  in.vec(D.lsk);
  in.vec(D.lsp);

  C = new_compact_core(G);
  C->load(in, G, ft.ctailOff);

  Borders::HLists& B = D.hLists;
  const int muMax = B.muMax;
  in.seek(ft.bdirOff);
  in.vec(B.off);
  in.vec(B.boff);
  in.vec(B.soff);
  B.WB.resize(size_t(B.off[muMax + 1]));
  B.LO.resize(B.WB.size());
  B.X.resize(B.WB.size());
  B.U.resize(B.WB.size());
  B.BM.resize(size_t(B.boff[muMax + 1]));
  B.ST.resize(size_t(B.soff[muMax + 1]));

  in.seek(ft.ctailOff + ft.ctailLen);
  while (in.tell() < ft.bdirOff) {
    in.next();
    uint64_t h[4];
    in.get(h, sizeof h);
    const int mu = int(h[0]);
    const int64_t sizeBHmu = B.off[mu + 1] - B.off[mu], nbl = B.boff[mu + 1] - B.boff[mu],
                  sizeST = B.soff[mu + 1] - B.soff[mu];
    in.get(B.WB.data() + B.off[mu], size_t(sizeBHmu) * 4);
    in.get(B.LO.data() + B.off[mu], size_t(sizeBHmu) * 4);
    in.get(B.X.data() + B.off[mu], size_t(sizeBHmu) * 4);
    in.get(B.U.data() + B.off[mu], size_t(sizeBHmu) * 4);
    in.get(B.BM.data() + B.boff[mu], size_t(nbl) * 4);
    in.get(B.ST.data() + B.soff[mu], size_t(sizeST) * 4);
  }

  fclose(in.f);
}
