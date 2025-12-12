import os
import pandas as pd
import matplotlib.pyplot as plt

def pareto_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["D_x"].notna() & df["B_x"].notna()].copy()
    df["D_x"] = df["D_x"].astype(float)
    df["B_x"] = df["B_x"].astype(float)

    df = df.sort_values(["D_x", "B_x"]).drop_duplicates(subset=["D_x", "B_x"], keep="first")

    # mantener frontera: B estrictamente decreciente al crecer D
    keep = []
    best_B = float("inf")
    for _, r in df.iterrows():
        if r["B_x"] < best_B - 1e-9:
            keep.append(True)
            best_B = r["B_x"]
        else:
            keep.append(False)
    return df.loc[keep].copy()

def plot_pareto(csv_path: str):
    df = pd.read_csv(csv_path)
    df = pareto_clean(df)

    out_dir = os.path.dirname(csv_path)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_png = os.path.join(out_dir, f"{base}_frontier.png")

    plt.figure()
    plt.plot(df["D_x"], df["B_x"], marker="o")
    plt.xlabel("D(x) logrado")
    plt.ylabel("B(x)")
    plt.title("Frontera Pareto (ε-constraint)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"✅ Guardado: {out_png}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python plot_pareto_frontier.py <pareto_csv_path>")
    plot_pareto(sys.argv[1])
