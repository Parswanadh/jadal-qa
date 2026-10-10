#!/usr/bin/env python3
"""Regenerate docs/assets/scoreboard.svg on ONE linear axis (finding 6)."""
X0, W, VMIN, VMAX = 290, 480, 0.3, 1.0

def x(v):
    return X0 + (v - VMIN) / (VMAX - VMIN) * W

rows = [
    # (label, sub, value, lo, hi, bar_color, pill_color, text_color, bold_label)
    ("in-domain \u00b7 ensemble \u2014 gates not met", "IndEgo \u00b7 n=150 \u00b7 7 clips",
     0.834, 0.715, 0.914, "#0f172a", "#0ea5e9", "#ffffff", True),
    ("in-domain \u00b7 S1 head", "IndEgo \u00b7 n=150",
     0.802, 0.710, 0.870, "#64748b", "#334155", "#ffffff", False),
    ("out-of-domain \u00b7 cascade", "factories \u00b7 n=160 \u00b7 47 sites \u00b7 sync doc; file pending",
     0.600, 0.474, 0.724, "#b45309", "#f59e0b", "#ffffff", True),
    ("out-of-domain \u00b7 ensemble", "factories \u00b7 n=160 \u00b7 sync doc; file pending",
     0.566, 0.433, 0.693, "#94a3b8", "#94a3b8", "#ffffff", False),
]
ys = [104, 150, 196, 240]
AXIS_Y = 254
chance_x = x(0.5)

parts = []
parts.append('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="312" viewBox="0 0 800 312" role="img" aria-label="Honest scoreboard: in-domain AUROC about 0.8 on IndEgo versus out-of-domain AUROC about 0.6 on factory footage, with confidence intervals">')
parts.append('  <rect x="0" y="0" width="800" height="312" rx="12" fill="#f8fafc"/>')
parts.append('  <text x="24" y="32" font-family="Inter, Helvetica, Arial, sans-serif" font-size="18" font-weight="700" fill="#0f172a">The honest scoreboard</text>')
parts.append('  <text x="24" y="52" font-family="Inter, Helvetica, Arial, sans-serif" font-size="12.5" fill="#475569">AUROC with 95% CI \u00b7 dashed line = chance (0.5) \u00b7 CIs bootstrap windows, optimistic \u00b7 linear axis 0.3\u20131.0</text>')
parts.append('  <!-- axis: single linear scale 0.3 to 1.0 -->')
parts.append(f'  <line x1="{X0}" y1="70" x2="{X0}" y2="{AXIS_Y}" stroke="#cbd5e1" stroke-width="1.5"/>')
parts.append(f'  <line x1="{X0}" y1="{AXIS_Y}" x2="{X0 + W}" y2="{AXIS_Y}" stroke="#cbd5e1" stroke-width="1.5"/>')
parts.append(f'  <line x1="{chance_x:.1f}" y1="70" x2="{chance_x:.1f}" y2="{AXIS_Y}" stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="5 4"/>')
for t in (0.3, 0.5, 0.7, 1.0):
    parts.append(f'  <text x="{x(t) - 12:.0f}" y="268" font-family="Inter, Helvetica, Arial, sans-serif" font-size="11" fill="#64748b">{t:.1f}</text>')

label_ys = [(98, 116), (144, 162), (190, 208), (234, 248)]
for (lab, sub, v, lo, hi, bar_c, pill_c, txt_c, bold), y, (ly1, ly2) in zip(rows, ys, label_ys):
    w = ' font-weight="700"' if bold else ''
    fill = "#0f172a" if bold else "#0f172a"
    parts.append(f'  <text x="24" y="{ly1}" font-family="Inter, Helvetica, Arial, sans-serif" font-size="12.5"{w} fill="{fill}">{lab}</text>')
    parts.append(f'  <text x="24" y="{ly2}" font-family="Inter, Helvetica, Arial, sans-serif" font-size="11.5" fill="#64748b">{sub}</text>')
    parts.append('  <g>')
    parts.append(f'    <line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" stroke="{bar_c}" stroke-width="3" stroke-linecap="round"/>')
    px = x(v) - 45
    parts.append(f'    <rect x="{px:.1f}" y="{y - 10}" width="90" height="20" rx="5" fill="{pill_c}"/>')
    parts.append(f'    <text x="{px + 7:.1f}" y="{y + 4}" font-family="monospace, monospace" font-size="12" font-weight="700" fill="{txt_c}">{v:.3f}</text>')
    parts.append('  </g>')

parts.append('  <text x="24" y="296" font-family="Inter, Helvetica, Arial, sans-serif" font-size="11.5" fill="#475569">Scores above chance in-domain on 7 clips (exploratory; weakest clip 0.606). Factory transfer is the open problem.</text>')
parts.append('</svg>')

svg = "\n".join(parts) + "\n"
with open("docs/assets/scoreboard.svg", "w") as f:
    f.write(svg)

print("wrote docs/assets/scoreboard.svg")
for lab, sub, v, lo, hi, *_ in rows:
    print(f"{lab}: lo={x(lo):.1f} val={x(v):.1f} hi={x(hi):.1f}")
print(f"chance(0.5)={chance_x:.1f}")
