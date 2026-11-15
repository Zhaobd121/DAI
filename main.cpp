#include "build.hpp"
#include <sys/stat.h>

static FILE* open_out(const string& path) {
  FILE* f = fopen(path.c_str(), "w");
  if (!f) die("cannot write " + path);
  return f;
}

static void make_dirs(const string& dir) {
  for (size_t i = 1; i <= dir.size(); i++)
    if (i == dir.size() || dir[i] == '/') {
      const string p = dir.substr(0, i);
      if (mkdir(p.c_str(), 0755) != 0 && errno != EEXIST) die("cannot create folder " + p);
    }
}

static string dataset_name(const string& path) {
  const size_t b = path.rfind('/') + 1;
  const string name = path.substr(b, path.find('.', b) - b);
  if (name.empty()) die("cannot take a dataset name from " + path);
  return name;
}

static string result_dir(const string& dataset) {
#ifdef VARIANT
  return "result/" + dataset + "/" VARIANT;
#else
  return "result/" + dataset;
#endif
}

struct Counts {
  long long memb = 0, cores = 0, clusters = 0;
};

static Counts write_answer(const Engine& eng, int n, const string& path) {
  const vector<pair<int, int>>& out = eng.out;
  const size_t nCore = eng.ncore;
  vector<int> minCore(size_t(n), INT_MAX);
  for (size_t i = 0; i < nCore; i++)
    minCore[out[i].second] = min(minCore[out[i].second], out[i].first);
  struct Row { int key, border, v; };
  vector<Row> rows(out.size());
  for (size_t i = 0; i < out.size(); i++) rows[i] = {minCore[out[i].second], i < nCore ? 0 : 1, out[i].first};
  sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
    return tie(a.key, a.border, a.v) < tie(b.key, b.border, b.v);
  });
  Counts c;
  c.memb = (long long)out.size();
  c.cores = (long long)nCore;
  vector<int32_t> body;
  body.reserve(out.size() + 3 * nCore);
  for (size_t i = 0; i < rows.size();) {
    size_t j = i, h = i;
    while (j < rows.size() && rows[j].key == rows[i].key) j++;
    while (h < j && rows[h].border == 0) h++;
    body.push_back(int32_t(c.clusters++));
    body.push_back(int32_t(h - i));
    body.push_back(int32_t(j - h));
    for (size_t r = i; r < j; r++) body.push_back(rows[r].v);
    i = j;
  }
  FILE* fa = open_out(path);
  const int64_t hd[2] = {c.memb, c.clusters};
  if (fwrite("DAIANSWR", 1, 8, fa) != 8 || fwrite(hd, 8, 2, fa) != 2 ||
      fwrite(body.data(), 4, body.size(), fa) != body.size() || fclose(fa) != 0)
    die("cannot write " + path);
  return c;
}

static pair<double, double> load_and_sim(const string& path, Graph& G) {
  const auto a = Clock::now();
  G = load_graph(path);
  const auto b = Clock::now();
  compute_similarity(G);
  return {secs(a, b), secs(b, Clock::now())};
}

static int parse_positive(const char* s) {
  char* end;
  const long long x = strtoll(s, &end, 10);
  if (*end != '\0' || x <= 0 || x > INT_MAX)
    die(string("not a positive integer: ") + s);
  return int(x);
}

static int usage() {
  fprintf(stderr,
          "usage: dai build <graph.txt>\n"
          "       dai query <dataset> <eps_num> <eps_den> <mu> [repeats]\n");
  return 1;
}

int main(int argc, char** argv) {
  set_malloc_policy();
  if (argc < 3) return usage();
  const string cmd = argv[1];
  if (cmd == "build") {
    const string dir = result_dir(dataset_name(argv[2]));
    make_dirs(dir);
    Graph G;
    const auto loadSimTimes = load_and_sim(argv[2], G);
    Engine eng(G);
    const BuildTimes bt = build_index_file(G, eng, dir + "/index.bin");

    FILE* ft = open_out(dir + "/times.tsv");
    fprintf(ft, "load_s\t%.6f\nbuild_s\t%.6f\nwrite_s\t%.6f\nn\t%d\nm\t%lld\nindex_bytes\t%llu\n", loadSimTimes.first,
            loadSimTimes.second + bt.raw - bt.ser, bt.ser, G.n, (long long)G.m, (unsigned long long)bt.total);
    fclose(ft);
    return 0;
  }

  if (cmd == "query" && argc >= 6) {
    const int epsNum = parse_positive(argv[3]), epsDen = parse_positive(argv[4]), mu = parse_positive(argv[5]);
    const int repeats = argc >= 7 ? parse_positive(argv[6]) : 7;
    if (epsNum > epsDen) die("eps must be at most 1");
    if (mu < 2) die("mu must be at least 2");
    const string dir = result_dir(dataset_name(argv[2]));
    Graph G;
    Engine eng(G);
    const auto a = Clock::now();
    load_index(dir + "/index.bin", G, eng.D, eng.CC);
    const double loadTime = secs(a, Clock::now());
    const string qdir = dir + "/eps_" + to_string(epsNum) + "-" + to_string(epsDen) + "_mu_" + to_string(mu);
    make_dirs(qdir);
    eng.prepare();

    const Threshold eps{uint64_t(epsNum), uint64_t(epsDen)};
    for (int w = 0; w < 2; w++) eng.query(eps, mu);
    vector<double> ms(repeats);
    for (double& t : ms) {
      const auto b = Clock::now();
      eng.query(eps, mu);
      t = chrono::duration<double, milli>(Clock::now() - b).count();
    }
    nth_element(ms.begin(), ms.begin() + repeats / 2, ms.end());
    const Counts c = write_answer(eng, G.n, qdir + "/answer.bin");

    FILE* fq = open_out(qdir + "/query.tsv");
    for (FILE* f : {stdout, fq}) {
      fprintf(f, "eps\t%d/%d\nmu\t%d\nindex_load_ms\t%.3f\nmedian_ms\t%.6f\n", epsNum, epsDen, mu, loadTime * 1e3,
              ms[repeats / 2]);
      fprintf(f, "memberships\t%lld\ncores\t%lld\nborders\t%lld\nclusters\t%lld\n", c.memb, c.cores,
              c.memb - c.cores, c.clusters);
    }
    fclose(fq);
    return 0;
  }
  return usage();
}
