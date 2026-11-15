#pragma once
#include <bits/stdc++.h>
#include <sys/mman.h>
#include <malloc.h>
#include <immintrin.h>
#include <unistd.h>
#include <fcntl.h>
using namespace std;
using Clock = chrono::steady_clock;
typedef unsigned __int128 u128;

static double secs(Clock::time_point a, Clock::time_point b) {
  return chrono::duration<double>(b - a).count();
}
static int bitsFor(uint64_t x) { return x ? 64 - __builtin_clzll(x) : 0; }
static inline uint64_t load8(const uint8_t* p) {
  uint64_t x;
  memcpy(&x, p, 8);
  return x;
}

[[noreturn]] static void die(const string& s) {
  fprintf(stderr, "dai: %s\n", s.c_str());
  exit(2);
}

static void set_malloc_policy() {
  mallopt(M_MMAP_THRESHOLD, 1 << 20);
  mallopt(M_TRIM_THRESHOLD, 1 << 20);
}
static void trim_heap() { malloc_trim(0); }

template <class T>
struct HugeAlloc {
  using value_type = T;
  HugeAlloc() = default;
  template <class U>
  HugeAlloc(const HugeAlloc<U>&) {}
  T* allocate(size_t k) {
    const size_t bytes = max<size_t>(1, k * sizeof(T)), H = size_t(2) << 20;
    void* p = nullptr;
    if (bytes >= 8 * H) {
      if (posix_memalign(&p, H, bytes)) throw std::bad_alloc();
      madvise(p, bytes & ~(H - 1), MADV_HUGEPAGE);
    } else {
      p = malloc(bytes);
      if (!p) throw std::bad_alloc();
    }
    return static_cast<T*>(p);
  }
  void deallocate(T* p, size_t) { free(p); }
  template <class U>
  void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(p)) U;
  }
  template <class U, class... A>
  void construct(U* p, A&&... a) {
    ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
  }
  template <class U>
  bool operator==(const HugeAlloc<U>&) const { return true; }
  template <class U>
  bool operator!=(const HugeAlloc<U>&) const { return false; }
};
template <class T>
using hvec = vector<T, HugeAlloc<T>>;
template <class T>
struct HugeBuf {
  T* p;
  explicit HugeBuf(size_t k) : p(HugeAlloc<T>().allocate(k)) {}
  HugeBuf(const HugeBuf&) = delete;
  HugeBuf& operator=(const HugeBuf&) = delete;
  ~HugeBuf() {
    HugeAlloc<T>().deallocate(p, 0);
  }
  T& operator[](size_t i) { return p[i]; }
};

struct Threshold { uint64_t p, q; };

static inline int cmp_cos2(uint64_t x, uint64_t y, const Threshold& eps) {
  const u128 l = u128(x) * (eps.q * eps.q), r = u128(eps.p * eps.p) * y;
  return l < r ? -1 : (l > r ? 1 : 0);
}

struct Edge { int u, v; };

template <class EU, class EV>
static vector<int> krt_leaf_order(int n, int64_t m, EU eu, EV ev) {
  struct KrtNode { int lc, rc, sz, off, root; };
  vector<KrtNode> kn(n);
  vector<int> p(n), rep(n);
  for (int x = 0; x < n; x++) {
    p[x] = x;
    rep[x] = ~x;
  }
  auto find_root = [&](int x) {
    while (p[x] != x) {
      p[x] = p[p[x]];
      x = p[x];
    }
    return x;
  };
  int nc = 0;
  for (int64_t q = 0; q < m; q++) {
    const int a = find_root(eu(q)), b = find_root(ev(q));
    if (a == b) continue;
    const int ra = rep[a], rb = rep[b];
    const int sa = ra >= 0 ? kn[ra].sz : 1, sb = rb >= 0 ? kn[rb].sz : 1;
    const int j = nc++;
    kn[j] = {ra, rb, sa + sb, 0, 1};
    if (ra >= 0) kn[ra].root = 0;
    if (rb >= 0) kn[rb].root = 0;
    if (sa < sb) {
      p[a] = b;
      rep[b] = j;
    } else {
      p[b] = a;
      rep[a] = j;
    }
  }

  vector<int> leafOrder(n);
  int base = 0;
  for (int q = nc - 1; q >= 0; q--) {
    KrtNode& z = kn[q];
    if (z.root) {
      z.off = base;
      base += z.sz;
    }
    const int o = z.off;
    int c = z.lc, leftSize;
    if (c >= 0) {
      kn[c].off = o;
      leftSize = kn[c].sz;
    } else {
      leafOrder[o] = ~c;
      leftSize = 1;
    }
    c = z.rc;
    if (c >= 0) kn[c].off = o + leftSize;
    else leafOrder[o + leftSize] = ~c;
  }

  vector<char> seen(n, 0);
  for (int q = 0; q < base; q++) seen[leafOrder[q]] = 1;
  for (int x = 0; x < n; x++)
    if (!seen[x]) leafOrder[base++] = x;
  return leafOrder;
}

static int spWidth(int64_t nb) {
  return nb <= 1 ? 1 : 32 - __builtin_clz(unsigned(nb - 1));
}
static int64_t spRow(int64_t nmb, int j) {
  return int64_t(j) * (nmb + 1) - ((int64_t(1) << j) - 1);
}
static int64_t spSize(int64_t nb) {
  const int width = spWidth(nb);
  if (!nb) return 0;
  const int64_t nmb = (nb + width - 1) / width;
  const int rows = 32 - __builtin_clz(unsigned(nmb));
  return nb + spRow(nmb, rows);
}
template <class T, bool kMin>
static inline bool spBetter(const T* a, int x, int y) {
  return kMin ? a[x] < a[y] : a[x] > a[y];
}
template <class T, bool kMin, class I>
static I treePick(const T* a, I x, I y) {
  if (spBetter<T, kMin>(a, x, y)) return x;
  if (spBetter<T, kMin>(a, y, x)) return y;
  return min(x, y);
}
template <class T, bool kMin, class I>
static void spBuild(const T* a, int nb, I* table) {
  const int width = spWidth(nb);
  const int nmb = int((int64_t(nb) + width - 1) / width);
  I* macro = table + nb;
  for (int block = 0; block < nmb; block++) {
    const int begin = int(int64_t(block) * width);
    const int end = begin + min(width, nb - begin);
    uint32_t mask = 0;
    for (int i = begin; i < end; i++) {
      while (mask) {
        const int q = begin + 31 - __builtin_clz(mask);
        if (!spBetter<T, kMin>(a, i, q)) break;
        mask &= ~(uint32_t(1) << (q - begin));
      }
      mask |= uint32_t(1) << (i - begin);
      table[i] = I(mask);
    }
    macro[block] = I(begin + __builtin_ctz(mask));
  }
  for (int j = 1; (int64_t(1) << j) <= nmb; j++) {
    I* row = macro + spRow(nmb, j);
    const I* prev = macro + spRow(nmb, j - 1);
    const int half = 1 << (j - 1);
    const int count = nmb - (1 << j) + 1;
    for (int i = 0; i < count; i++)
      row[i] = treePick<T, kMin>(a, prev[i], prev[i + half]);
  }
}
template <class I>
static inline int spInside(const I* table, int width, int l, int r) {
  const int begin = (l / width) * width;
  const uint32_t mask = uint32_t(table[r]) & (~uint32_t(0) << (l - begin));
  return begin + __builtin_ctz(mask);
}
template <class T, bool kMin, class I>
struct SpTable {
  const T* a;
  const I* table;
  int nb, width, nmb;
  SpTable(const T* a_, int nb_, const I* table_)
      : a(a_), table(table_), nb(nb_), width(spWidth(nb_)), nmb(int((int64_t(nb_) + width - 1) / width)) {}
  int query(int l, int r) const {
    if (l == r) return l;
    const int bl = l / width, br = r / width;
    if (bl == br) return spInside(table, width, l, r);
    int best = treePick<T, kMin>(a, spInside(table, width, l, (bl + 1) * width - 1),
                                 spInside(table, width, br * width, r));
    const int first = bl + 1, last = br - 1;
    if (first <= last) {
      const int j = 31 - __builtin_clz(unsigned(last - first + 1));
      const I* row = table + nb + spRow(nmb, j);
      const I middle = treePick<T, kMin>(a, row[first], row[last - (1 << j) + 1]);
      best = treePick<T, kMin>(a, best, int(middle));
    }
    return best;
  }
};

struct BlockOut {
  static constexpr size_t kBuf = size_t(64) << 20;
  int fd = -1;
  string path;
  vector<char> buf;
  size_t bn = 0;
  uint32_t nb = 0;
  uint64_t total = 16;
  double sum = 0;
  struct Span {
    BlockOut& o;
    Clock::time_point a;
    explicit Span(BlockOut& o_) : o(o_), a(Clock::now()) {}
    ~Span() { o.sum += secs(a, Clock::now()); }
  };
  void flush_buf() {
    for (size_t k = 0; k < bn;) {
      const ssize_t w = ::write(fd, buf.data() + k, bn - k);
      if (w <= 0) die("cannot write " + path);
      k += size_t(w);
    }
    bn = 0;
  }
  void raw(const void* p, size_t len) {
    const char* s = static_cast<const char*>(p);
    while (len) {
      const size_t k = min(len, kBuf - bn);
      memcpy(buf.data() + bn, s, k);
      bn += k;
      s += k;
      len -= k;
      if (bn == kBuf) flush_buf();
    }
  }
  void put(const void* p, size_t len) {
    Span s(*this);
    raw(p, len);
  }
  void begin(const char* name, uint64_t len) {
    char h[24];
    memset(h, 0, 24);
    strncpy(h, name, 15);
    memcpy(h + 16, &len, 8);
    put(h, 24);
    nb++;
    total += 24 + len;
  }
  template <class T, class A>
  void vec(const char* name, const vector<T, A>& v) {
    begin(name, v.size() * sizeof(T));
    put(v.data(), v.size() * sizeof(T));
  }
  void field4(const void* base, size_t stride, size_t count) {
    Span s(*this);
    const char* p = static_cast<const char*>(base);
    for (size_t i = 0; i < count; i++) {
      if (bn + 4 > kBuf) flush_buf();
      memcpy(buf.data() + bn, p + i * stride, 4);
      bn += 4;
    }
  }
};
