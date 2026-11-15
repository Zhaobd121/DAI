# DAI

This repository provides the C++ implementation of DAI from the paper

Bide Zhao, Lijun Chang, and Xin Huang. DAI: A Degree-Adaptive Index for Efficient and Exact Structural Graph
Clustering. Submitted to ICDE 2027.

It contains DAI and the three variants of DAI used in the experiments of the paper:

- **DAI** (source: `main.cpp` and `*.hpp`) - the index of the paper.
- **-KRT** (patch: `ablation/minus_KRT.patch`) - core clusters are found by union-find instead of KRTs.
- **-lo** (patch: `ablation/minus_lo.patch`) - queries use neither the lo values nor their range-minimum structure.
- **-C** (patch: `ablation/minus_C.patch`) - at each level mu, the core forest and KRT are built from all edges
  between vertices of degree at least mu - 1, not from the reduced edge set.

Each variant is a copy of the source code of DAI with one patch applied. All four programs follow the same workflow:
1) **Build** the index of a graph
2) **Query** the index at any eps and mu

---

## Prerequisites

- Linux on x86-64
- `g++` with C++17
- `make` and `patch`

---

## Project Layout

- `main.cpp` and `*.hpp` - the source code of DAI
- `Makefile` - builds DAI and the variants
- `ablation/` - one patch per variant

---

## Compile

```bash
make
```

This compiles DAI and the three variants. To compile one of them only, give its name to `make`:

| Program | Command | Executable |
|---|---|---|
| DAI | `make dai` | `dai` |
| -KRT | `make minus_KRT` | `ablation/minus_KRT/dai` |
| -lo | `make minus_lo` | `ablation/minus_lo/dai` |
| -C | `make minus_C` | `ablation/minus_C/dai` |

Each variant is compiled from its own copy of the source code with its patch applied, kept in its folder under
`ablation`, so the source code of DAI is not changed. After `make`, the variants can be run without compiling
again; run `make` again only after the source code or a patch is changed. `make clean` deletes the four
executables and the variant folders.

---

## Input Format

Example format:

```txt
<u_1> <v_1>
<u_2> <v_2>
...
```

The input is a text file. Each undirected edge is given on one line by the ids of its two end vertices.

- Vertex ids are integers starting from 0. The number of vertices is the largest id plus 1, so an id that does
  not appear is a vertex without edges.
- Each edge is listed once, as `u v` or `v u`, and there are no self-loops; otherwise `build` stops with an error.
- The two ids are separated by spaces or tabs. Lines that do not start with two integers, such as comments
  starting with `#` or `%`, are skipped.

For example, this file is a triangle 0-1-2 and the edge 2-3:

```txt
# a triangle 0-1-2 and the edge 2-3
0 1
0 2
1 2
2 3
```

The SNAP file `com-amazon.ungraph.txt` is also in this format and can be used as it is.

---

## Usage

The programs keep all their outputs in the folder `result` under the current folder, one folder per dataset, and
create missing folders themselves.

### 1) Build the index

```bash
./dai build <graph.txt>
```

- `graph.txt`: the input graph

The name of the dataset is the file name of the graph up to its first dot, for example `com-amazon` for
`com-amazon.ungraph.txt`. The index is saved to `result/<dataset>/index.bin`. Graphs with the same dataset name
share this folder, so a new build replaces the old index.

### 2) Query

```bash
./dai query <dataset> <eps_num> <eps_den> <mu> [repeats]
```

- `dataset`: the name of a dataset whose index has been built
- `eps_num`, `eps_den`: eps = eps_num / eps_den, given by two positive integers with eps_num <= eps_den
- `mu`: an integer of at least 2
- `repeats`: the number of timed runs, 7 by default; the query is first run twice to warm up

The answer is saved to the folder `result/<dataset>/eps_<eps_num>-<eps_den>_mu_<mu>`.

Here eps is the threshold on the structural similarity defined in the paper and mu the size threshold: a vertex is
a core if at least mu vertices among itself and its neighbors have similarity at least eps with it.

### Example

```bash
./dai build graph.txt
./dai query graph 7 20 4
```

The first command builds the index of the dataset `graph`. The second one queries it at eps = 7/20 = 0.35 and
mu = 4 and saves the answer to `result/graph/eps_7-20_mu_4`.

### Variants

The variants are run in the same way with their own executables, which `make` has already compiled, so no
compiling is needed. Each variant keeps its outputs in a folder of its own inside the folder of the dataset,
such as `result/graph/minus_KRT` for -KRT:

```bash
./ablation/minus_KRT/dai build graph.txt
./ablation/minus_KRT/dai query graph 7 20 4
```

---

## Output

After the commands above, the folder `result` holds the following files:

```txt
result/
  graph/
    index.bin              the index of DAI
    times.tsv              the lines saved by build
    eps_7-20_mu_4/
      answer.bin           the answer
      query.tsv            the lines printed by query
    minus_KRT/
      index.bin            the same files for -KRT
      times.tsv
      eps_7-20_mu_4/
        answer.bin
        query.tsv
```

`build` prints nothing and saves one line `name<TAB>value` for each of the following to `times.tsv`, in this order:

| Name | Meaning | Unit |
|---|---|---|
| `load_s` | the time to read the graph | seconds |
| `build_s` | the time to compute the similarities and build the index, not counting the time to write it | seconds |
| `write_s` | the time to write `index.bin` | seconds |
| `n` | the number of vertices, that is, the largest vertex id plus 1 | - |
| `m` | the number of edges | - |
| `index_bytes` | the size of `index.bin` | bytes |

`query` prints one line `name<TAB>value` for each of the following and saves the same lines to `query.tsv`:

| Name | Meaning | Unit |
|---|---|---|
| `eps` | eps as a fraction, such as `7/20` | - |
| `mu` | mu | - |
| `index_load_ms` | the time to load the index | milliseconds |
| `median_ms` | the median time of the timed runs | milliseconds |
| `memberships` | the number of (vertex, cluster) pairs in the answer | - |
| `cores` | the number of cores | - |
| `borders` | the number of (vertex, cluster) pairs whose vertex is not a core | - |
| `clusters` | the number of clusters | - |

`answer.bin` stores the clusters one after another. It starts with the 8 bytes `DAIANSWR` and two little-endian
int64: the total number of vertices in all clusters, where a vertex is counted once for each cluster it is in (equal
to `memberships`), and the number of clusters (equal to `clusters`). Then each cluster is stored as the following
little-endian int32, in this order:

| Field | Meaning |
|---|---|
| cluster id | 0, 1, 2, ... in the order in which the clusters are stored |
| `a` | the number of cores in the cluster |
| `b` | the number of other vertices in the cluster |
| `a` vertex ids | the cores of the cluster, in increasing order |
| `b` vertex ids | the other vertices of the cluster, in increasing order |

The clusters are stored in increasing order of their smallest core id. Each core is in exactly one cluster, while a
vertex that is not a core can be in several clusters. A vertex that is in no cluster does not appear in the file.
