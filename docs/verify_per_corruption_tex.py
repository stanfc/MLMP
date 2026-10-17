"""Verify docs/per_corruption_round150_tables_best_second.tex against the RAW results files (independent of the md table)."""
import os, re, sys
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")
TEX = "docs/per_corruption_round150_tables_best_second.tex"
SETS = [("ACDC", "ACDCDataset", "acdc"), ("Cityscapes-C", "CityscapesDataset", "cityscapes"), ("VOC20-C", "PascalVOC20Dataset", "voc20"),
        ("VOC21-C", "PascalVOC21Dataset", "voc21"), ("Pascal-Context59-C", "PascalContext59Dataset", "pc59"),
        ("Pascal-Context60-C", "PascalContext60Dataset", "pc60"), ("COCO-Stuff-C", "COCOStuffDataset", "coco-stuff"), ("COCO-Object-C", "COCOObjectDataset", "coco-object")]
STD = {"TENT": "tent_continual", "SAR": "sar_continual", "SPA": "spa_continual", "TRIBE": "tribe_continual", "DeYO": "deyo_continual",
       "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual", "MGP": "mgp_deyo_mlmp_continual", "DAF": "daf_mlmp_continual"}
COLS = ["No-Adapt", "MLMP-episodic", "TENT", "SAR", "SPA", "TRIBE", "DeYO", "ViDA", "PTTA", "COME", "MGP", "DAF", "CAGA"]
def find(root, b):
    for s in (f"{b}_15corr_sub100", f"{b}_sub100_15corr", b):
        if os.path.isdir(f"{root}/{s}"): return f"{root}/{s}"
def last_round(p, expect150=True):
    hdr, rows = None, {}
    for l in open(p + "/results_all_rounds.txt"):
        if l.startswith("Round,"): hdr = [x.strip() for x in l.split(",")[1:]]
        elif l.startswith("Round "):
            q = [x.strip() for x in l.split(",")]; rows[int(q[0].split()[1])] = [float(x) for x in q[1:]]
    r = max(rows)
    if expect150: assert r == 150, (p, r)
    assert len(hdr) == len(rows[r]), (p, hdr)
    return dict(zip(hdr, rows[r]))
def episodic(p):
    d = {}
    for l in list(open(p + "/results.txt"))[1:]:
        q = l.split(",")
        try: d[q[0].strip()] = float(q[1].split("+/-")[0])
        except (ValueError, IndexError): pass
    d["Mean_mIoU"] = sum(d.values()) / len(d); return d
def raw(folder):
    root = f"save/{folder}"; a = folder == "ACDCDataset"; c = {}
    c["No-Adapt"] = last_round(f"{root}/naclip_no_adapt" if a else find(root, "no_adapt"), False)
    c["MLMP-episodic"] = episodic(f"{root}/naclip_mlmp_episodic" if a else find(root, "mlmp_episodic"))
    for k, v in STD.items(): c[k] = last_round(f"{root}/naclip_tent_continual" if (a and k == "TENT") else (f"{root}/{v}" if a else find(root, v)))
    c["CAGA"] = last_round(f"{root}/adagate_hgate" if a else find(root, "adagate_hgate")); return c
def parse_cell(s):
    s = s.strip(); tag = None
    m = re.fullmatch(r"\\textbf\{(.+)\}", s)
    if m: tag, s = "best", m.group(1)
    m = re.fullmatch(r"\\underline\{(.+)\}", s)
    if m: tag, s = "second", m.group(1)
    return float(s), tag
tex = open(TEX).read(); errs = []; nchk = 0
tabs = re.findall(r"\\begin\{table\*\}.*?\\end\{table\*\}", tex, re.S)
if len(tabs) != 8: errs.append(f"expected 8 tables, found {len(tabs)}")
if "RDumb" in tex.split("\\begin{table*}")[1]: errs.append("RDumb appears in table")
for (title, folder, tag), tab in zip(SETS, tabs):
    if f"tab:per-corruption-{tag}" not in tab: errs.append(f"{title}: label mismatch")
    if title not in tab.split("\\label")[0]: errs.append(f"{title}: caption does not name dataset")
    raw_ = raw(folder)
    hdr = re.search(r"Corruption &(.*?)\\\\\s*\n\\midrule", tab, re.S).group(1)
    hdr = re.sub(r"\\shortstack\{([^\\}]*)\\\\([^}]*)\}", r"\1\2", hdr)
    names = [h.replace("\\textbf{", "").replace("}", "").strip() for h in hdr.split("&")]
    if names != COLS: errs.append(f"{title}: header columns {names} != {COLS}")
    ncols = re.search(r"\\begin\{tabular\}\{(\w+)\}", tab).group(1)
    if len(ncols) != 14: errs.append(f"{title}: tabular spec has {len(ncols)} cols")
    body = tab.split("\\midrule", 1)[1]
    rows = [r for r in re.split(r"\\\\\s*\n", body) if "&" in r]
    conds_raw = [c for c in raw_["CAGA"] if c != "Mean_mIoU"]
    data_rows = [r for r in rows if not r.strip().startswith(("\\midrule", "\\textbf{Average}")) and "Average" not in r]
    if len(data_rows) != len(conds_raw): errs.append(f"{title}: {len(data_rows)} corruption rows vs {len(conds_raw)} in raw")
    all_rows = [(conds_raw[i] if i < len(conds_raw) else None, r) for i, r in enumerate(data_rows)] + [("Mean_mIoU", [r for r in rows if "Average" in r][0])]
    avg_check = {n: [] for n in COLS}
    for key, r in all_rows:
        r = r.replace("\\midrule", "").strip(); cells = [x for x in r.split("&")]
        label = cells[0].replace("\\textbf{", "").replace("}", "").strip()
        if key != "Mean_mIoU" and label.lower().replace(" ", "_") != key.lower(): errs.append(f"{title}: row label '{label}' vs raw '{key}'")
        vals = []
        for n, cell in zip(COLS, cells[1:]):
            v, t = parse_cell(cell); vals.append((v, t)); nchk += 1
            rv = raw_[n][key]
            if abs(round(rv, 2) - v) > 1e-9: errs.append(f"{title}/{label}/{n}: tex {v} != raw {rv:.4f}")
            if key != "Mean_mIoU": avg_check[n].append(rv)
        rvals = [round(raw_[n][key], 2) for n in COLS]
        dist = sorted(set(rvals), reverse=True); best, second = dist[0], (dist[1] if len(dist) > 1 else None)
        for n, (v, t) in zip(COLS, vals):
            exp = "best" if v == best else ("second" if v == second else None)
            if t != exp: errs.append(f"{title}/{label}/{n}: marker {t} but expected {exp} (value {v}, best {best}, second {second})")
    for n in COLS:
        m = sum(avg_check[n]) / len(avg_check[n])
        if abs(m - raw_[n]["Mean_mIoU"]) > 0.006: errs.append(f"{title}/{n}: mean of rows {m:.3f} != reported Mean_mIoU {raw_[n]['Mean_mIoU']:.3f}")
print(f"tables={len(tabs)} cells checked={nchk}")
print("ERRORS:" if errs else "ALL CHECKS PASSED"); [print(" -", e) for e in errs]
