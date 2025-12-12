import os
from pathlib import Path
import io
import sys
import pandas as pd
import numpy as np

DATA_DIR = Path(r"/mongo/data/development_analysis")
OUT_DIR = Path(__file__).parent / "summary_output"


def read_csv_safely(path: Path) -> pd.DataFrame:
    # Try common variants
    tries = [
        dict(sep=',', engine='c', encoding='utf-8'),
        dict(sep=';', engine='c', encoding='utf-8'),
        dict(sep=',', engine='python', encoding='utf-8'),
        dict(sep=';', engine='python', encoding='utf-8'),
        dict(sep=',', engine='c', encoding='utf-8-sig'),
        dict(sep=';', engine='c', encoding='utf-8-sig'),
    ]
    last_err = None
    for kw in tries:
        try:
            return pd.read_csv(path, **kw)
        except Exception as e:
            last_err = e
    raise last_err


def df_info_to_str(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def basic_stats(df: pd.DataFrame) -> dict:
    num = df.select_dtypes(include=[np.number])
    desc = num.describe().T if not num.empty else pd.DataFrame()
    missing = df.isna().sum().sort_values(ascending=False)
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    dup_rows = int(df.duplicated().sum())
    unique_counts = df.nunique(dropna=False)
    return {
        'numeric_cols': list(num.columns),
        'describe_num': desc,
        'missing': missing,
        'missing_pct': missing_pct,
        'duplicates': dup_rows,
        'unique_counts': unique_counts,
    }


def quick_opinion(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    num = df.select_dtypes(include=[np.number])
    num_cols = len(num.columns)
    # candidate target = last numeric column
    target = num.columns[-1] if num_cols >= 1 else None
    issues = []
    lr_ok = True
    rf_ok = True

    # Row count heuristic
    if rows < 200:
        issues.append(f"Low sample size ({rows} rows).")
        # still possible but caution

    # Numeric features heuristic
    if num_cols < 2:
        issues.append("Not enough numeric columns for regression (need at least 2: features + target).")
        lr_ok = False
        rf_ok = False

    # Missingness heuristic
    miss_ratio = df.isna().mean().mean()
    if miss_ratio > 0.3:
        issues.append(f"High overall missingness (~{miss_ratio*100:.1f}%). Consider imputation or feature removal.")

    # Target variance
    if target is not None:
        var = float(np.nanvar(num[target]))
        if var == 0.0:
            issues.append(f"Target '{target}' has zero variance.")
            lr_ok = False
            rf_ok = False

    # Linearity expectation for LR
    if num_cols >= 2 and rows >= 200:
        # quick check of some linear correlations
        try:
            corr = num.corr(numeric_only=True).abs()
            if target is not None and target in corr.columns:
                best = corr[target].drop(labels=[target]).max() if corr.shape[1] > 1 else 0.0
                if best < 0.1:
                    issues.append("Weak linear correlation between features and target — LR may underperform; RF still fine.")
                    # LR still okay but with caveats
        except Exception:
            pass

    verdict_lr = "Yes" if lr_ok else ("With caveats" if num_cols >= 2 else "No")
    verdict_rf = "Yes" if rf_ok else "No"

    note = "; ".join(issues) if issues else "No major issues detected for a baseline run."
    return (
        f"Linear Regression suitability: {verdict_lr}\n"
        f"Random Forest suitability: {verdict_rf}\n"
        f"Notes: {note}"
    )


def summarize_dataset(csv_path: Path) -> str:
    df = read_csv_safely(csv_path)
    lines = []
    lines.append(f"=== {csv_path.name} ===")
    lines.append(f"Path: {csv_path}")
    lines.append(f"Shape: {df.shape}")
    lines.append("")

    # Info
    lines.append("[df.info]")
    lines.append(df_info_to_str(df).rstrip())
    lines.append("")

    # Head
    lines.append("[head(5)]")
    try:
        lines.append(df.head().to_string(max_cols=30, max_rows=5))
    except Exception:
        lines.append(str(df.head()))
    lines.append("")

    # Missingness & stats
    stats = basic_stats(df)
    lines.append("[missing by column]")
    lines.append(stats['missing'].to_string())
    lines.append("")
    lines.append("[missing % by column]")
    lines.append(stats['missing_pct'].map(lambda x: f"{x:.2f}%").to_string())
    lines.append("")

    if not stats['describe_num'].empty:
        lines.append("[numeric describe]")
        lines.append(stats['describe_num'].to_string())
        lines.append("")
    else:
        lines.append("[numeric describe] No numeric columns found.")
        lines.append("")

    lines.append(f"Duplicate rows: {stats['duplicates']}")
    lines.append("")

    # Opinion
    lines.append("[model suitability opinion]")
    lines.append(quick_opinion(df))
    lines.append("")

    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    csvs = sorted([p for p in DATA_DIR.glob('*.csv')])
    if not csvs:
        print(f"No CSV files found in {DATA_DIR}")
        sys.exit(0)

    report_path = OUT_DIR / "data_summary.countries_of_interest.txt"
    all_lines = []
    all_lines.append("DATA SUMMARY REPORT")
    all_lines.append(f"Folder: {DATA_DIR}")
    all_lines.append("")

    for csv in csvs:
        try:
            all_lines.append(summarize_dataset(csv))
        except Exception as e:
            all_lines.append(f"=== {csv.name} ===")
            all_lines.append(f"Error reading file: {e}")
            all_lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_lines))

    print("Summary written to:", report_path)


if __name__ == "__main__":
    main()
