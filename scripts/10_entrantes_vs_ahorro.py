"""
Relación entre fracción de contenedores entrantes (RECV+DSCH / I0+RECV+DSCH)
y ahorro de distancia del modelo respecto a la operación real.

Hipótesis: a mayor fracción entrante, más control tiene el modelo sobre la
ubicación de contenedores y mayor es el ahorro esperado.
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR  = Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
RESULTADOS  = ROOT / "resultados" / "pipeline_bahia_criterio_iii_7d_theta1.4_alfa2_umbral20_90_base"
INST_DIR    = RESULTADOS / "instancias_coloracion"
ANALISIS    = SCRIPT_DIR / "analisis_resultados_dist.xlsx"
OUT_FIG_DIR = SCRIPT_DIR / "09_figuras"
OUT_FIG_DIR.mkdir(exist_ok=True)

# ── distancias (reducción % por semana) ───────────────────────────────────

real   = pd.read_excel(ANALISIS, sheet_name="Distancias reales")
modelo = pd.read_excel(ANALISIS, sheet_name="Distancias modelo").rename(columns={
    "Distancia Total": "Distancia_Total",
    "Distancia LOAD":  "Distancia_LOAD",
    "Distancia DLVR":  "Distancia_DLVR",
})

df = real[["Semana","Distancia_LOAD","Distancia_DLVR","Distancia_YARD"]].merge(
    modelo[["Semana","Distancia_Total","Distancia_LOAD","Distancia_DLVR"]],
    on="Semana", suffixes=("_real","_modelo")
)
df["dist_real_sin_yard"] = df["Distancia_LOAD_real"] + df["Distancia_DLVR_real"]
df["ahorro_pct"] = (
    (df["dist_real_sin_yard"] - df["Distancia_Total"]) / df["dist_real_sin_yard"] * 100
).round(2)

# ── fracción de contenedores entrantes por semana ─────────────────────────

rows = []
for semana in df["Semana"]:
    inst_path = INST_DIR / semana / f"Instancia_{semana}_K.xlsx"
    if not inst_path.exists():
        continue
    xl   = pd.ExcelFile(inst_path)
    dp   = xl.parse("D_params")
    i0   = xl.parse("I0_sb")

    recv  = dp["DR"].sum()
    dsch  = dp["DD"].sum()
    i0_total = i0["I0"].sum()
    total = recv + dsch + i0_total

    rows.append({
        "Semana":       semana,
        "recv":         recv,
        "dsch":         dsch,
        "i0":           i0_total,
        "total":        total,
        "frac_entrante": (recv + dsch) / total * 100 if total > 0 else 0,
    })

flujos = pd.DataFrame(rows)
data   = df.merge(flujos, on="Semana")

# ── scatter + regresión ───────────────────────────────────────────────────

import numpy as np

x = data["frac_entrante"].values
y = data["ahorro_pct"].values

coef = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
y_line = np.polyval(coef, x_line)

corr = np.corrcoef(x, y)[0, 1]

fig, ax = plt.subplots(figsize=(7, 5))

# colorear por signo de ahorro
colors = ["#d62728" if yi < 0 else "#1f77b4" for yi in y]
ax.scatter(x, y, c=colors, alpha=0.75, s=40, zorder=3)

ax.plot(x_line, y_line, color="gray", linewidth=1.2, linestyle="--",
        label=f"Tendencia (r = {corr:.2f})")

ax.axhline(0, color="black", linewidth=0.7, linestyle="-")
ax.set_xlabel("Contenedores entrantes / total (\\%)\n[RECV + DSCH] / [I0 + RECV + DSCH]")
ax.set_ylabel("Ahorro de distancia (\\%)\n[dist\\_real $-$ dist\\_modelo] / dist\\_real")
ax.legend(fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)

# marcar semanas 3 y 12
for label, semana, offset in [("Sem. 3", "2022-01-17", (0.5, 2)),
                               ("Sem. 12", "2022-03-21", (0.5, -4))]:
    row = data[data["Semana"] == semana]
    if not row.empty:
        xi, yi = row["frac_entrante"].values[0], row["ahorro_pct"].values[0]
        ax.annotate(label, xy=(xi, yi), xytext=(xi + offset[0], yi + offset[1]),
                    fontsize=8, arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

plt.tight_layout()
out_pdf = OUT_FIG_DIR / "entrantes_vs_ahorro.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Figura PDF: {out_pdf}")
print(f"Correlación (r): {corr:.3f}")
print(f"Pendiente: {coef[0]:.2f} pp ahorro por pp de fracción entrante")

# ── LaTeX / pgfplots ──────────────────────────────────────────────────────

coords_pos = " ".join(
    f"({row.frac_entrante:.2f},{row.ahorro_pct:.2f})"
    for row in data.itertuples() if row.ahorro_pct >= 0
)
coords_neg = " ".join(
    f"({row.frac_entrante:.2f},{row.ahorro_pct:.2f})"
    for row in data.itertuples() if row.ahorro_pct < 0
)

x0, x1 = float(x.min()), float(x.max())
y0_line = float(np.polyval(coef, x0))
y1_line = float(np.polyval(coef, x1))

# semanas anotadas
sem3  = data[data["Semana"] == "2022-01-17"].iloc[0]
sem12 = data[data["Semana"] == "2022-03-21"].iloc[0]

tex = rf"""
\begin{{figure}}[htbp]
\centering
\begin{{tikzpicture}}
\begin{{axis}}[
  width=0.82\textwidth, height=6cm,
  xlabel={{Fracción de contenedores entrantes [\%] (RECV$+$DSCH) / (I0$+$RECV$+$DSCH)}},
  ylabel={{Ahorro de distancia [\%]}},
  xmin=28, xmax=76,
  ymajorgrids=true, xmajorgrids=true,
  grid style={{dashed, gray!35}},
  legend style={{at={{(0.05,0.95)}}, anchor=north west, font=\small}},
]
% semanas con ahorro positivo
\addplot[only marks, mark=*, mark size=1.8pt, blue!60, fill opacity=0.7]
  coordinates {{{coords_pos}}};
% semanas con ahorro negativo
\addplot[only marks, mark=*, mark size=1.8pt, red!70, fill opacity=0.7]
  coordinates {{{coords_neg}}};
% línea de tendencia
\addplot[dashed, gray, thick]
  coordinates {{({x0:.2f},{y0_line:.2f}) ({x1:.2f},{y1_line:.2f})}};
\addlegendentry{{Ahorro $>0$}}
\addlegendentry{{Ahorro $<0$}}
\addlegendentry{{Tendencia ($r={corr:.2f}$)}}
% línea horizontal en 0
\addplot[black, thin] coordinates {{(28,0) (76,0)}};
% anotaciones semanas extremas
\node[font=\footnotesize, anchor=west] at (axis cs:{sem3.frac_entrante+0.8:.2f},{sem3.ahorro_pct-2:.2f}) {{Sem.~3}};
\node[font=\footnotesize, anchor=west] at (axis cs:{sem12.frac_entrante+0.8:.2f},{sem12.ahorro_pct+1.5:.2f}) {{Sem.~12}};
\end{{axis}}
\end{{tikzpicture}}
\caption{{Relación entre la fracción de contenedores entrantes y el ahorro de distancia
del modelo respecto a la operación real (52 semanas). Puntos azules: semanas con
reducción positiva; puntos rojos: semanas en que el modelo supera a la operación real en distancia.}}
\label{{fig:entrantes-vs-ahorro}}
\end{{figure}}
"""

out_tex = OUT_FIG_DIR / "entrantes_vs_ahorro.tex"
out_tex.write_text(tex.strip(), encoding="utf-8")
print(f"Figura LaTeX: {out_tex}")
print()
print(data[["Semana","frac_entrante","ahorro_pct","i0","recv","dsch"]].sort_values("frac_entrante").to_string(index=False))
