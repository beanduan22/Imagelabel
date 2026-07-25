"""Compute the paper statistics from the annotation files.

    python analyze.py --study ./study --a1 alice --a2 bob --adj carol

Reports, per model-method combination and overall:
  - cases adjudicated as preserving the source label (n, proportion),
  - 95% Wilson confidence interval,
  - Verified DoF: distinct alternative predicted classes among preserved cases,
and Cohen's kappa computed from the two primary annotators BEFORE adjudication.

Also writes <study>/validation_summary.csv. Stdlib only.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

Z = 1.959963984540054  # 95%


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f'Missing file: {path}')
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def latest_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rec in load_jsonl(path):
        out[rec['case_id']] = rec['label']
    return out


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cohen_kappa(l1: dict[str, str], l2: dict[str, str], ids: list[str]) -> tuple[float, int]:
    common = [cid for cid in ids if cid in l1 and cid in l2]
    n = len(common)
    if n == 0:
        return float('nan'), 0
    po = sum(1 for c in common if l1[c] == l2[c]) / n
    p1_yes = sum(1 for c in common if l1[c] == 'preserved') / n
    p2_yes = sum(1 for c in common if l2[c] == 'preserved') / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)
    kappa = 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
    return kappa, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='./study')
    ap.add_argument('--annotations', default='./annotations')
    ap.add_argument('--a1', required=True, help='primary annotator 1 id')
    ap.add_argument('--a2', required=True, help='primary annotator 2 id')
    ap.add_argument('--adj', default=None, help='adjudicator id (omit if no disagreements)')
    args = ap.parse_args()

    study = Path(args.study)
    ann = Path(args.annotations)
    cases = {r['case_id']: r for r in load_jsonl(study / 'cases.jsonl')}
    ids = sorted(cases)

    l1 = latest_labels(ann / f'{args.a1}.jsonl')
    l2 = latest_labels(ann / f'{args.a2}.jsonl')
    ladj = latest_labels(ann / f'{args.adj}.jsonl') if args.adj else {}

    for name, lab in ((args.a1, l1), (args.a2, l2)):
        missing = [c for c in ids if c not in lab]
        if missing:
            print(f'WARNING: {name} has {len(missing)} unlabeled case(s); '
                  'they are excluded from all statistics.')

    kappa, n_kappa = cohen_kappa(l1, l2, ids)

    # Adjudicated final label per case.
    final: dict[str, str] = {}
    unresolved = 0
    for cid in ids:
        if cid not in l1 or cid not in l2:
            continue
        if l1[cid] == l2[cid]:
            final[cid] = l1[cid]
        elif cid in ladj:
            final[cid] = ladj[cid]
        else:
            unresolved += 1
    if unresolved:
        print(f'WARNING: {unresolved} disagreement(s) lack an adjudicator label; excluded.')

    combos: dict[tuple[str, str, str], list[str]] = {}
    for cid in final:
        c = cases[cid]
        combos.setdefault((c['dataset'], c['model'], c['method']), []).append(cid)

    rows = []
    for key in sorted(combos):
        cids = combos[key]
        n = len(cids)
        preserved = [c for c in cids if final[c] == 'preserved']
        k = len(preserved)
        lo, hi = wilson(k, n)
        dof = len({cases[c].get('pred_class_index') for c in preserved})
        rows.append({
            'dataset': key[0], 'model': key[1], 'method': key[2],
            'n': n, 'preserved': k, 'proportion': round(k / n, 4) if n else 0,
            'wilson_lo': round(lo, 4), 'wilson_hi': round(hi, 4),
            'verified_dof': dof,
        })

    n_all = len(final)
    k_all = sum(1 for c in final if final[c] == 'preserved')
    lo, hi = wilson(k_all, n_all)

    hdr = f"{'dataset':<12}{'model':<12}{'method':<14}{'n':>4}{'pres.':>7}{'prop.':>8}" \
          f"{'95% Wilson CI':>18}{'DoF':>6}"
    print()
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['dataset']:<12}{r['model']:<12}{r['method']:<14}{r['n']:>4}"
              f"{r['preserved']:>7}{r['proportion']:>8.3f}"
              f"   [{r['wilson_lo']:.3f}, {r['wilson_hi']:.3f}]{r['verified_dof']:>6}")
    print('-' * len(hdr))
    print(f"{'ALL':<38}{n_all:>4}{k_all:>7}{(k_all / n_all if n_all else 0):>8.3f}"
          f"   [{lo:.3f}, {hi:.3f}]")
    print(f"\nCohen's kappa ({args.a1} vs {args.a2}, pre-adjudication, "
          f"n={n_kappa}): {kappa:.4f}")

    out = study / 'validation_summary.csv'
    with out.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ['dataset', 'model', 'method', 'n', 'preserved',
                            'proportion', 'wilson_lo', 'wilson_hi', 'verified_dof'])
        w.writeheader()
        w.writerows(rows)
    print(f'Written: {out}')


if __name__ == '__main__':
    main()
