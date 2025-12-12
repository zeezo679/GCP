import random

n = 30
density = 0.8  # 0.0–1.0, where 1.0 = complete graph

edges = []

for u in range(1, n + 1):
    for v in range(u + 1, n + 1):
        if random.random() < density:
            edges.append((u, v))

print(f"Generated {len(edges)} edges")
for u, v in edges:
    print(u, v)
