import csv
import matplotlib.pyplot as plt

# Simple CSV reader: columns: prec,fuse,tm,tn,tk,ms
rows = []
with open('autotune_results.csv','r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# group by (prec,fuse)
from collections import defaultdict
groups = defaultdict(list)
for r in rows:
    key = (r['prec'], r['fuse'])
    ms = float(r['ms'])
    tm = int(r['tm'])
    tn = int(r['tn'])
    tk = int(r['tk'])
    groups[key].append((tm,tn,tk,ms))

for key,vals in groups.items():
    vals.sort(key=lambda x: x[3])
    best = vals[0]
    print(f"prec={key[0]} fuse={key[1]} best tile=({best[0]},{best[1]},{best[2]}) ms={best[3]:.6f}")

# produce a simple bar chart of best ms per (prec,fuse)
labels = []
msvals = []
for key,vals in groups.items():
    best = min(vals, key=lambda x: x[3])
    labels.append(f"p{key[0]}_f{key[1]}")
    msvals.append(best[3])

plt.figure(figsize=(8,4))
plt.bar(labels, msvals)
plt.ylabel('ms')
plt.title('Autotune best ms per (prec,fuse)')
plt.tight_layout()
plt.savefig('autotune_summary.png')
print('Saved autotune_summary.png')
