"""Download all Poker44 benchmark releases to /home/sn126/data/benchmark/."""
import json
import time
import urllib.request
from pathlib import Path

BASE = "https://api.poker44.net/api/v1/benchmark"
OUT = Path("/home/sn126/data/benchmark")
OUT.mkdir(parents=True, exist_ok=True)


def get(url: str):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())["data"]
        except Exception as e:
            print(f"retry {attempt+1} after error: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


releases = get(f"{BASE}/releases?limit=100")["releases"]
print(f"{len(releases)} releases", flush=True)

for rel in releases:
    date = rel["sourceDate"]
    out_file = OUT / f"chunks_{date}.json"
    if out_file.exists():
        print(f"{date}: already downloaded", flush=True)
        continue
    all_chunks = []
    cursor = None
    while True:
        url = f"{BASE}/chunks?sourceDate={date}&limit=24"
        if cursor:
            url += f"&cursor={cursor}"
        data = get(url)
        all_chunks.extend(data["chunks"])
        cursor = data.get("nextCursor")
        if not cursor:
            break
    out_file.write_text(json.dumps({"release": rel, "chunks": all_chunks}))
    n_groups = sum(len(c.get("chunks", [])) for c in all_chunks)
    print(f"{date}: {len(all_chunks)} chunks, {n_groups} groups saved", flush=True)

print("DONE", flush=True)
