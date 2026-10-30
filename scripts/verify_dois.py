"""Check every DOI cited in the README against the Crossref API and record the
registered title/year, so citations can be audited (results/dois.json)."""
import json
import re
import urllib.request

from _common import ROOT, save_json

readme = open(f"{ROOT}/README.md", encoding="utf-8").read()


def _clean(d):
    """Strip trailing punctuation and unbalanced ')' left by markdown links."""
    while d and (d[-1] in ".,;:" or (d[-1] == ")" and d.count(")") > d.count("("))):
        d = d[:-1]
    return d


dois = sorted(set(_clean(d) for d in re.findall(r"10\.\d{4,9}/[^\s\]>\"']+", readme)))
out = {}
for d in dois:
    url = "https://api.crossref.org/works/" + urllib.request.quote(d)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "taulapse-doi-check (mailto:maggiolotommaso@gmail.com)"})
        with urllib.request.urlopen(req, timeout=30) as r:
            m = json.load(r)["message"]
        year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
        out[d] = {"ok": True, "title": (m.get("title") or [""])[0], "year": year,
                  "container": (m.get("container-title") or [""])[0]}
    except Exception as e:  # noqa: BLE001
        out[d] = {"ok": False, "error": str(e)}
    print(("OK  " if out[d]["ok"] else "FAIL"), d, "|", out[d].get("year"), out[d].get("title", out[d].get("error"))[:90])
save_json("dois.json", out)
