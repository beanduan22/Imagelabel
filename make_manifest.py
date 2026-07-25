"""Export sampled prediction-change cases from a LATTE failures .pt into the
annotation manifest (cases.jsonl) + PNG images.

Run once per (model, method) combination, appending to the same manifest:

  python make_manifest.py \
      --failures ../Latte/results/mnist_lenet5_single/failures_single.pt \
      --dataset MNIST --model lenet5 --method latte \
      --num 30 --seed 0 --out ./study

For baseline methods with a different storage format, write your own exporter
that emits the same JSONL fields (see README.md); the GUI only reads the
manifest, never the .pt files.

Case ids are random hex so that neither the id nor the image path leaks the
generation method to annotators.
"""
from __future__ import annotations
import argparse
import json
import secrets
from pathlib import Path

import torch
from torchvision.utils import save_image

MNIST_CLASSES = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck',
]


def load_class_info(dataset: str, path: str | None) -> dict:
    """Return {class_index: {label, synonyms, definition}}."""
    if path:
        raw = json.loads(Path(path).read_text())
        return {int(k): v for k, v in raw.items()}
    name = dataset.lower()
    if name == 'mnist':
        names = MNIST_CLASSES
    elif name in ('cifar10', 'cifar-10'):
        names = CIFAR10_CLASSES
    else:
        raise SystemExit(
            f'No builtin class names for {dataset}; pass --class-info JSON '
            '({"0": {"label": ..., "synonyms": [...], "definition": ...}, ...})'
        )
    return {i: {'label': n, 'synonyms': [], 'definition': ''} for i, n in enumerate(names)}


def collect_failures(result: dict) -> list[dict]:
    cases = []
    for seed in result['seed_results']:
        for f in seed['failures']:
            if 'x' not in f or 'x_seed' not in f:
                raise SystemExit(
                    'Failures were saved without images (store_samples: false); '
                    're-run LATTE with latte.store_samples: true.'
                )
            cases.append({**f, 'seed_class': seed['seed_class']})
    return cases


def denorm(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'half':
        x = x * 0.5 + 0.5
    return x.clamp(0.0, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--failures', required=True)
    ap.add_argument('--dataset', required=True, help='e.g. MNIST, CIFAR10, ImageNet')
    ap.add_argument('--model', required=True, help='e.g. lenet5, vgg16 (hidden from annotators)')
    ap.add_argument('--method', required=True, help='e.g. latte, baselineX (hidden from annotators)')
    ap.add_argument('--num', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='./study', help='output dir (holds cases.jsonl + images/)')
    ap.add_argument('--class-info', default=None,
                    help='JSON: class index -> {label, synonyms, definition}; required for ImageNet')
    ap.add_argument('--normalization', default='half', choices=['half', 'none'])
    args = ap.parse_args()

    out_dir = Path(args.out)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / 'cases.jsonl'

    class_info = load_class_info(args.dataset, args.class_info)
    result = torch.load(args.failures, map_location='cpu')
    pool = collect_failures(result)
    if len(pool) < args.num:
        raise SystemExit(f'Only {len(pool)} prediction-change cases available, need {args.num}.')

    gen = torch.Generator().manual_seed(args.seed)
    picked = [pool[i] for i in torch.randperm(len(pool), generator=gen)[:args.num].tolist()]

    with manifest.open('a') as fh:
        for case in picked:
            cid = secrets.token_hex(6)
            src_rel = f'images/{cid}_a.png'
            gen_rel = f'images/{cid}_b.png'
            save_image(denorm(case['x_seed'], args.normalization), out_dir / src_rel)
            save_image(denorm(case['x'], args.normalization), out_dir / gen_rel)
            gt = class_info.get(int(case['og_a']), {'label': str(case['og_a']),
                                                    'synonyms': [], 'definition': ''})
            row = {
                'case_id': cid,
                'dataset': args.dataset,
                'model': args.model,
                'method': args.method,
                'source_image': src_rel,
                'generated_image': gen_rel,
                'gt_class_index': int(case['og_a']),
                'gt_label': gt['label'],
                'gt_synonyms': gt.get('synonyms', []),
                'gt_definition': gt.get('definition', ''),
                'pred_class_index': int(case['pred_a']),
                'seed_idx': int(case['seed_idx']),
            }
            fh.write(json.dumps(row) + '\n')

    print(f'Appended {len(picked)} cases ({args.dataset}/{args.model}/{args.method}) '
          f'to {manifest}')


if __name__ == '__main__':
    main()
