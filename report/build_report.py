"""Assemble report/backbone_report.html from the source template + live results.

Kept in the repo, not the session scratchpad: the scratchpad lives in /tmp and was
wiped by the 2026-09-04 19:11 reboot, taking the previous copy of this with it.

  python report/build_report.py
then publish report/backbone_report.html as the artifact.
"""
import base64, csv, os, pathlib

HERE = pathlib.Path(__file__).parent
FIG = pathlib.Path("figures/backbone")
ROOT = "save/ACDCDataset"
ABL, SMOKE = f"{ROOT}/backbone_ablation", f"{ROOT}/seed_variance"
SMOKE = f"{ROOT}/backbone_smoke"
TOTAL = 150


def b64(name):
    return "data:image/png;base64," + base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()


def miou(d):
    p = os.path.join(d, "results_all_rounds.txt")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        parts = [q.strip() for q in line.strip().split(",")]
        if len(parts) < 2 or not parts[0].lower().startswith("round "):
            continue
        try:
            out.append((int(parts[0].split()[1]), float(parts[-1])))
        except (ValueError, IndexError):
            pass
    out.sort()
    return [v for _, v in out]


def firing(d):
    p = os.path.join(d, "gate_log.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return sum(1 for r in rows if float(r["rst"]) > 0) / len(rows) if rows else None


ROWS = [
    ("NA-CLIP  ViT-L/14", f"{ROOT}/batch_ablation/deyo_mlmp_b1",
     f"{ROOT}/batch_ablation/gradnorm_scaled_b1", f"{SMOKE}/noadapt_naclip-ViT-L_14", True),
    ("ClearCLIP k-k  ViT-L/14", f"{ABL}/clearclip_L14_nogate", f"{ABL}/clearclip_L14_gate",
     f"{SMOKE}/noadapt_clearclip-ViT-L_14", False),
    ("ClearCLIP q-q  ViT-L/14", f"{ABL}/clearqq_L14_nogate", f"{ABL}/clearqq_L14_gate",
     f"{SMOKE}/noadapt_clearclip_qq-ViT-L_14", False),
    ("MaskCLIP  ViT-L/14", f"{ABL}/maskclip_L14_nogate", f"{ABL}/maskclip_L14_gate",
     f"{SMOKE}/noadapt_maskclip-ViT-L_14", False),
    ("v-v attention  ViT-L/14", f"{ABL}/vvclip_L14_nogate", f"{ABL}/vvclip_L14_gate",
     f"{SMOKE}/noadapt_vvclip-ViT-L_14", False),
    ("NA-CLIP gauss-only  ViT-L/14", f"{ABL}/nonly_L14_nogate", f"{ABL}/nonly_L14_gate",
     f"{SMOKE}/noadapt_naclip_nonly-ViT-L_14", False),
    ("NA-CLIP  ViT-B/16", f"{ABL}/naclip_B16_nogate", f"{ABL}/naclip_B16_gate",
     f"{SMOKE}/noadapt_naclip-ViT-B_16", False),
    ("ClearCLIP k-k  ViT-B/16", f"{ABL}/clearclip_B16_nogate", f"{ABL}/clearclip_B16_gate",
     f"{SMOKE}/noadapt_clearclip-ViT-B_16", False),
    ("NA-CLIP  ViT-B/32", f"{ABL}/naclip_B32_nogate", f"{ABL}/naclip_B32_gate",
     f"{SMOKE}/noadapt_naclip-ViT-B_32", False),
]

trs, rail = [], []
for name, p_no, p_ga, p_fl, is_ref in ROWS:
    fl, no, ga = miou(p_fl), miou(p_no), miou(p_ga)
    f = firing(p_ga)
    n = min(len(no), len(ga))
    floor = f"{fl[0]:.2f}" if fl else "&mdash;"
    if n == 0:
        started = max(len(no), len(ga))
        pill = ('<span class="pill live">running</span>' if started
                else '<span class="pill wait">queued</span>')
        cells = ['<td class="dim">&mdash;</td>'] * 4
    else:
        d = ga[n - 1] - no[n - 1]
        done = n >= TOTAL
        pill = ('<span class="pill done">R150 complete</span>' if done
                else f'<span class="pill live">R{n} of {TOTAL}</span>')
        dcell = f"{d:+.2f}" + ("" if done else f' <span class="dim">@R{n}</span>')
        cells = [f"<td>{no[n-1]:.2f}</td>", f"<td>{ga[n-1]:.2f}</td>",
                 f'<td class="up">{dcell}</td>' if d > 0.5 else f"<td>{dcell}</td>",
                 f"<td>{f:.3f}</td>" if f is not None else '<td class="dim">&mdash;</td>']
    trs.append(f'<tr class="{"ref" if is_ref else ""}"><td class="name">{name}'
               + (' <span class="dim">(ref)</span>' if is_ref else '')
               + f'</td><td>{floor}</td>' + "".join(cells) + f"<td>{pill}</td></tr>")
    pct = min(100, round(100 * n / TOTAL))
    rail.append(f'<div><span class="who">{name}</span><span class="st">{pill}</span>'
                f'<span class="bar"><i class="{"full" if pct >= 100 else ""}" '
                f'style="width:{pct}%"></i></span></div>')

src = (HERE / "backbone_report_src.html").read_text()
out = (src.replace("__IMG_FLOOR__", b64("backbone_floor"))
          .replace("__IMG_TRAJ__", b64("backbone_trajectories"))
          .replace("__IMG_TRANSFER__", b64("backbone_transfer"))
          .replace("__TABLE_ROWS__", "\n        ".join(trs))
          .replace("__RAIL__", "\n    ".join(rail)))
dest = HERE / "backbone_report.html"
dest.write_text(out)
print(f"wrote {dest}  ({len(out)/1024/1024:.2f} MB, {len(ROWS)} rows)")
