"""Download the two open datasets used in the real-data section.

Data are NOT redistributed in this repository; this script fetches them from
their canonical hosts into ./data and records SHA-256 checksums.

1. HYPERAKTIV (Hicks et al. 2021, doi:10.1145/3458305.3478454), OSF project 3agwr.
   Adult ADHD patients vs *clinical* (non-ADHD psychiatric) controls; trial-level
   Conners' CPT-II. OSF metadata license: CC-BY 4.0; the project wiki adds
   "competitions and commercial purposes need prior written permission".
2. OpenNeuro ds000102 "Flanker task (event-related)" (Kelly et al. 2008),
   healthy adults, license PDDL (public domain). Only the *_events.tsv files.
"""
import hashlib
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

FILES = {
    "hyperaktiv/CPT_II_ConnersContinuousPerformanceTest.csv": "https://osf.io/download/2gt3w/",
    "hyperaktiv/patient_info.csv": "https://osf.io/download/hypj6/",
    "ds000102/dataset_description.json": "https://s3.amazonaws.com/openneuro.org/ds000102/dataset_description.json",
    "ds000102/participants.tsv": "https://s3.amazonaws.com/openneuro.org/ds000102/participants.tsv",
}
for s in range(1, 27):
    for r in (1, 2):
        FILES[f"ds000102/sub-{s:02d}_run-{r}_events.tsv"] = (
            f"https://s3.amazonaws.com/openneuro.org/ds000102/sub-{s:02d}/func/"
            f"sub-{s:02d}_task-flanker_run-{r}_events.tsv")


def main():
    sums = {}
    for rel, url in FILES.items():
        dst = os.path.join(DATA, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(dst, "wb") as f:
                    f.write(r.read())
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {rel}: {e}", file=sys.stderr)
                continue
        with open(dst, "rb") as f:
            sums[rel] = hashlib.sha256(f.read()).hexdigest()
        print(f"ok {rel}")
    with open(os.path.join(DATA, "SHA256SUMS.json"), "w") as f:
        json.dump(sums, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
