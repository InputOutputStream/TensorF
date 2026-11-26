#!/usr/bin/env python3
"""Write a synthetic dataset with the exact OGB node-property layout (raw/*.csv.gz, split/<scheme>/*.csv.gz),
so the converter/loader and download_datasets.sh --from can be exercised without network access.
  scripts/make_fake_ogb.py OUT_ZIP [N=3000] [F=32] [C=6] [avg_deg=8]
The graph has planted communities (edges prefer same-class nodes) and class-dependent features, so a GNN can learn it."""
import gzip, io, os, random, sys, zipfile
out = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
F = int(sys.argv[3]) if len(sys.argv) > 3 else 32; C = int(sys.argv[4]) if len(sys.argv) > 4 else 6
deg = int(sys.argv[5]) if len(sys.argv) > 5 else 8
rnd = random.Random(7); lab = [rnd.randrange(C) for _ in range(N)]
by = [[i for i in range(N) if lab[i] == c] for c in range(C)]
def gz(rows): return gzip.compress(("\n".join(rows) + "\n").encode())
edges = []
for u in range(N):
    for _ in range(deg // 2):
        v = rnd.choice(by[lab[u]]) if rnd.random() < 0.8 else rnd.randrange(N)
        if v != u: edges.append("%d,%d" % (u, v))
feat = ["%s" % ",".join("%.4f" % (rnd.gauss(1.0 if (k % C) == lab[i] else 0.0, 0.7)) for k in range(F)) for i in range(N)]
idx = list(range(N)); rnd.shuffle(idx); a, b = int(N * .5), int(N * .75)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("fake/raw/edge.csv.gz", gz(edges)); z.writestr("fake/raw/node-feat.csv.gz", gz(feat))
    z.writestr("fake/raw/node-label.csv.gz", gz(str(x) for x in lab))
    z.writestr("fake/raw/num-node-list.csv.gz", gz([str(N)])); z.writestr("fake/raw/num-edge-list.csv.gz", gz([str(len(edges))]))
    for name, part in (("train", idx[:a]), ("valid", idx[a:b]), ("test", idx[b:])):
        z.writestr("fake/split/time/%s.csv.gz" % name, gz(str(x) for x in sorted(part)))
print("wrote %s: N=%d F=%d C=%d E=%d" % (out, N, F, C, len(edges)))
