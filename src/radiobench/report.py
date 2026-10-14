"""Results table (markdown) and plots from results.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = ["clean", "channel", "snr20", "snr10", "snr5", "snr0"]


def _lines(df: pd.DataFrame, col: str, title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    conds = [c for c in ORDER if c in set(df["condition"])]
    for system, g in df.groupby("system"):
        g = g.set_index("condition").reindex(conds)
        ax.plot(conds, g[col], marker="o", label=system)
    ax.set_title(title)
    ax.set_xlabel("condition (SNR falls to the right)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_report(results_csv: str | Path) -> Path:
    results_csv = Path(results_csv)
    out = results_csv.parent
    df = pd.read_csv(results_csv)
    radio = df[df["source"] == "radio"]
    _lines(radio, "wer", "WER on simulated radio audio", "WER", out / "wer_vs_condition.png")
    _lines(radio, "kw_recall", "Alert-phrase recall", "recall", out / "keyword_recall.png")
    _lines(radio, "false_alerts_per_hour", "False alerts per hour of audio", "false alerts / hour",
           out / "false_alerts.png")
    md = ["# Results", "", "## Radio set: WER / keyword recall / false alerts per hour", ""]
    piv = radio.assign(cell=lambda d: d.apply(
        lambda r: f"{r.wer:.1%} / {r.kw_recall:.0%} / {r.false_alerts_per_hour:.1f}", axis=1)
    ).pivot(index="system", columns="condition", values="cell")
    conds = [c for c in ORDER if c in piv.columns]
    md.append(piv[conds].to_markdown())
    ls = df[df["source"] == "librispeech"]
    if len(ls):
        md += ["", "## LibriSpeech test-clean sanity check (WER)", "",
               ls.set_index("system")[["wer"]].to_markdown(floatfmt=".3f")]
    md += ["", "## Speed (radio set, mean over conditions)", "",
           radio.groupby("system")[["rtf", "latency_s"]].mean().to_markdown(floatfmt=".2f")]
    (out / "results.md").write_text("\n".join(md), encoding="utf-8")
    return out / "results.md"
