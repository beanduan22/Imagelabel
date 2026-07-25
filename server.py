"""Blinded annotation GUI for the human semantic-validation study.

Primary annotation phase (annotators label independently):

    python server.py --study ./study --port 8765

Adjudication phase (third annotator sees only the disagreements):

    python server.py --study ./study --port 8765 --adjudicate alice bob

Protocol enforced by the server:
  - generation method / model / model prediction are never sent to the client;
  - case ids and image paths are opaque hex, so nothing leaks the combination;
  - case order is shuffled deterministically per annotator;
  - every judgment is appended immediately to annotations/<annotator>.jsonl
    (last record per case wins), so sessions can be resumed at any time.

Stdlib only — no dependencies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

STUDY_DIR = Path('./study')
ANN_DIR = Path('./annotations')
CASES: dict[str, dict] = {}
CASE_IDS: list[str] = []
MODE = 'primary'  # or 'adjudicate'

# Fields safe to expose to annotators (everything else stays server-side).
PUBLIC_FIELDS = ('case_id', 'dataset', 'gt_label', 'gt_synonyms', 'gt_definition')


def valid_annotator(name: str) -> bool:
    return bool(name) and name.replace('_', '').replace('-', '').isalnum()


def load_cases(manifest: Path) -> dict[str, dict]:
    cases = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            cases[row['case_id']] = row
    if not cases:
        raise SystemExit(f'No cases in {manifest}')
    return cases


def load_labels(annotator: str) -> dict[str, dict]:
    """case_id -> latest record for this annotator."""
    path = ANN_DIR / f'{annotator}.jsonl'
    labels: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                labels[rec['case_id']] = rec
    return labels


def append_label(rec: dict) -> None:
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    with (ANN_DIR / f"{rec['annotator']}.jsonl").open('a') as fh:
        fh.write(json.dumps(rec) + '\n')


def disagreement_ids(a1: str, a2: str) -> list[str]:
    l1, l2 = load_labels(a1), load_labels(a2)
    ids = []
    for cid in CASE_IDS:
        r1, r2 = l1.get(cid), l2.get(cid)
        if r1 is None or r2 is None:
            raise SystemExit(
                f'Case {cid} is missing a label from {a1 if r1 is None else a2}; '
                'both primary annotators must finish before adjudication.'
            )
        if r1['label'] != r2['label']:
            ids.append(cid)
    return ids


def annotator_order(annotator: str, ids: list[str]) -> list[str]:
    seed = int(hashlib.sha256(annotator.encode()).hexdigest(), 16) % (2 ** 32)
    order = list(ids)
    random.Random(seed).shuffle(order)
    return order


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), 'application/json')

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == '/':
            self._send(200, PAGE.encode(), 'text/html; charset=utf-8')
            return
        if path.startswith('/img/'):
            parts = path.split('/')  # /img/<case_id>/<a|b>
            if len(parts) == 4 and parts[2] in CASES and parts[3] in ('a', 'b'):
                case = CASES[parts[2]]
                key = 'source_image' if parts[3] == 'a' else 'generated_image'
                fp = STUDY_DIR / case[key]
                if fp.is_file():
                    self._send(200, fp.read_bytes(), 'image/png')
                    return
        self._json({'error': 'not found'}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length) or b'{}')
        except (ValueError, json.JSONDecodeError):
            self._json({'error': 'bad request body'}, 400)
            return
        if path == '/api/session':
            self._session(body)
        elif path == '/api/label':
            self._label(body)
        else:
            self._json({'error': 'not found'}, 404)

    def _session(self, body: dict) -> None:
        name = str(body.get('annotator', '')).strip()
        if not valid_annotator(name):
            self._json({'error': 'annotator id must be alphanumeric (plus - _)'}, 400)
            return
        order = annotator_order(name, CASE_IDS)
        labels = load_labels(name)
        cases = [{k: CASES[cid].get(k, '') for k in PUBLIC_FIELDS} for cid in order]
        self._json({
            'annotator': name,
            'mode': MODE,
            'cases': cases,
            'labels': {cid: {'label': r['label'], 'comment': r.get('comment', '')}
                       for cid, r in labels.items() if cid in CASES},
        })

    def _label(self, body: dict) -> None:
        name = str(body.get('annotator', '')).strip()
        cid = body.get('case_id')
        label = body.get('label')
        if not valid_annotator(name) or cid not in CASES \
                or label not in ('preserved', 'not_preserved'):
            self._json({'error': 'invalid label payload'}, 400)
            return
        append_label({
            'annotator': name,
            'role': 'adjudicator' if MODE == 'adjudicate' else 'primary',
            'case_id': cid,
            'label': label,
            'comment': str(body.get('comment', ''))[:2000],
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })
        self._json({'ok': True})


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Validation Study</title>
<style>
  :root {
    --bg: #f5f3ee; --panel: #ffffff; --ink: #1c1a17; --muted: #6f6a61;
    --line: #e3ded4; --accent: #1f6f54; --accent-ink: #ffffff;
    --no: #a33d2f; --radius: 10px;
    font-size: clamp(15px, 0.9rem + 0.2vw, 17px);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font-family: "Iowan Old Style", "Palatino Linotype", Georgia, ui-serif, serif;
    min-height: 100vh; display: flex; flex-direction: column;
  }
  header {
    display: flex; align-items: baseline; gap: 1rem; padding: 0.9rem 1.4rem;
    border-bottom: 1px solid var(--line); background: var(--panel);
  }
  header h1 { font-size: 1.05rem; margin: 0; letter-spacing: 0.02em; font-weight: 600; }
  header .who { color: var(--muted); font-size: 0.85rem; }
  header .spacer { flex: 1; }
  .progress { font-variant-numeric: tabular-nums; font-size: 0.85rem; color: var(--muted); }
  .bar { height: 4px; background: var(--line); }
  .bar > div { height: 100%; background: var(--accent); width: 0; transition: width 200ms ease; }

  main { flex: 1; display: grid; place-items: center; padding: 1.2rem; }
  .card {
    background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
    box-shadow: 0 12px 32px rgba(28,26,23,0.07);
    max-width: 60rem; width: 100%; padding: 1.6rem 2rem 1.8rem;
  }

  /* login */
  #login { max-width: 26rem; text-align: center; }
  #login h2 { margin: 0 0 0.4rem; font-size: 1.3rem; }
  #login p { color: var(--muted); margin: 0 0 1.2rem; font-size: 0.9rem; line-height: 1.5; }
  #login input {
    width: 100%; padding: 0.65rem 0.8rem; font: inherit; border: 1px solid var(--line);
    border-radius: 8px; margin-bottom: 0.8rem; background: var(--bg);
  }
  #login input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
  #login button { width: 100%; }
  .err { color: var(--no); font-size: 0.85rem; min-height: 1.2em; margin-top: 0.4rem; }

  /* task */
  .images { display: flex; gap: 2rem; justify-content: center; margin: 0.6rem 0 1rem; flex-wrap: wrap; }
  figure { margin: 0; text-align: center; }
  figure img {
    width: min(19rem, 42vw); aspect-ratio: 1; object-fit: contain;
    image-rendering: pixelated; background: #fff;
    border: 1px solid var(--line); border-radius: 6px;
  }
  figcaption {
    font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin-top: 0.5rem;
  }
  .labelbox {
    border-top: 1px solid var(--line); border-bottom: 1px solid var(--line);
    padding: 0.8rem 0.2rem; margin-bottom: 1.1rem; line-height: 1.5;
  }
  .labelbox .gt { font-size: 1.15rem; font-weight: 650; }
  .labelbox .aux { color: var(--muted); font-size: 0.88rem; }
  .question { text-align: center; font-size: 1.02rem; margin-bottom: 0.9rem; }
  .choices { display: flex; gap: 0.8rem; justify-content: center; flex-wrap: wrap; }
  button {
    font: inherit; cursor: pointer; border-radius: 8px; padding: 0.6rem 1.4rem;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    transition: transform 120ms ease, box-shadow 120ms ease;
  }
  button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(28,26,23,0.12); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  button:active { transform: translateY(0); }
  .btn-yes { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
  .btn-no  { background: var(--no); border-color: var(--no); color: #fff; }
  .btn-yes.picked, .btn-no.picked { box-shadow: 0 0 0 3px var(--bg), 0 0 0 5px currentColor; }
  kbd {
    font-family: ui-monospace, monospace; font-size: 0.75em; padding: 0.05em 0.4em;
    border: 1px solid rgba(255,255,255,0.5); border-radius: 4px; margin-left: 0.5em;
  }
  .nav { display: flex; justify-content: space-between; align-items: center; margin-top: 1.2rem; }
  .nav button { border: none; background: none; color: var(--muted); padding: 0.3rem 0.5rem; }
  .nav button:hover { color: var(--ink); box-shadow: none; }
  #comment {
    width: 100%; font: inherit; font-size: 0.88rem; border: 1px solid var(--line);
    border-radius: 8px; padding: 0.5rem 0.7rem; margin-top: 0.9rem; background: var(--bg);
    resize: vertical; min-height: 2.2rem;
  }
  #done { text-align: center; }
  #done h2 { color: var(--accent); }
  .hint { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 0.9rem; }
  .hidden { display: none !important; }
</style>
</head>
<body>
<header>
  <h1>Semantic Validation</h1>
  <span class="who" id="who"></span>
  <span class="spacer"></span>
  <span class="progress" id="progress"></span>
</header>
<div class="bar"><div id="barfill"></div></div>
<main>
  <div class="card" id="login">
    <h2>Annotator sign-in</h2>
    <p>Enter your assigned annotator ID. Your progress is saved after every
       judgment — you can close the tab and return at any time.<br>
       输入分配给你的标注员 ID。每次判断都会立刻保存,可随时中断继续。</p>
    <input id="name" placeholder="annotator ID, e.g. alice" autocomplete="off">
    <button class="btn-yes" id="start">Start / 开始</button>
    <div class="err" id="loginerr"></div>
  </div>

  <div class="card hidden" id="task">
    <div class="images">
      <figure><img id="img-src" alt="source image"><figcaption>Source · 原图</figcaption></figure>
      <figure><img id="img-gen" alt="generated image"><figcaption>Generated · 生成图</figcaption></figure>
    </div>
    <div class="labelbox">
      <div class="gt" id="gt"></div>
      <div class="aux" id="syn"></div>
      <div class="aux" id="def"></div>
    </div>
    <div class="question">
      Does the <strong>generated</strong> image still depict the source label above?<br>
      <span style="color:var(--muted);font-size:0.9rem">生成图是否仍然表现上述源标签的内容?</span>
    </div>
    <div class="choices">
      <button class="btn-yes" id="yes">Yes, label preserved · 保留<kbd>1</kbd></button>
      <button class="btn-no" id="no">No, not preserved · 未保留<kbd>2</kbd></button>
    </div>
    <textarea id="comment" placeholder="Optional comment · 备注(可选)"></textarea>
    <div class="nav">
      <button id="prev">&larr; Previous · 上一题</button>
      <button id="jump">Next unlabeled · 跳到未标注 &raquo;</button>
      <button id="next">Next · 下一题 &rarr;</button>
    </div>
    <div class="hint">Keyboard: <b>1</b> preserved · <b>2</b> not preserved · <b>&larr;/&rarr;</b> navigate</div>
  </div>

  <div class="card hidden" id="done">
    <h2>All cases labeled — thank you! · 全部完成,感谢!</h2>
    <p class="hint">You may still go back and revise any judgment.<br>仍可返回修改任意判断。</p>
    <div class="choices"><button id="review">Review from start · 从头检查</button></div>
  </div>
</main>
<script>
'use strict';
const $ = id => document.getElementById(id);
let S = null; // {annotator, cases, labels, idx}

async function api(path, body) {
  const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                               body: JSON.stringify(body)});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

$('start').onclick = async () => {
  const name = $('name').value.trim();
  $('loginerr').textContent = '';
  try {
    const s = await api('/api/session', {annotator: name});
    S = {annotator: s.annotator, mode: s.mode, cases: s.cases, labels: s.labels, idx: 0};
    const first = firstUnlabeled();
    S.idx = first === -1 ? 0 : first;
    $('login').classList.add('hidden');
    $('who').textContent = s.annotator + (s.mode === 'adjudicate' ? ' · adjudication' : '');
    render();
  } catch (e) { $('loginerr').textContent = e.message; }
};
$('name').addEventListener('keydown', e => { if (e.key === 'Enter') $('start').click(); });

function firstUnlabeled(from = 0) {
  for (let i = from; i < S.cases.length; i++)
    if (!S.labels[S.cases[i].case_id]) return i;
  for (let i = 0; i < from; i++)
    if (!S.labels[S.cases[i].case_id]) return i;
  return -1;
}
function labeledCount() {
  return S.cases.filter(c => S.labels[c.case_id]).length;
}

function render() {
  const n = S.cases.length, k = labeledCount();
  $('progress').textContent = k + ' / ' + n + ' labeled';
  $('barfill').style.width = (100 * k / n) + '%';
  if (k === n && S.showDone !== false) {
    $('task').classList.add('hidden'); $('done').classList.remove('hidden');
    return;
  }
  $('done').classList.add('hidden'); $('task').classList.remove('hidden');
  const c = S.cases[S.idx];
  $('img-src').src = '/img/' + c.case_id + '/a';
  $('img-gen').src = '/img/' + c.case_id + '/b';
  $('gt').textContent = 'Source label · 源标签: ' + c.gt_label +
      (c.dataset ? '   (' + c.dataset + ')' : '');
  $('syn').textContent = (c.gt_synonyms && c.gt_synonyms.length)
      ? 'Also called · 同义词: ' + c.gt_synonyms.join(', ') : '';
  $('def').textContent = c.gt_definition ? 'Definition · 释义: ' + c.gt_definition : '';
  const rec = S.labels[c.case_id];
  $('yes').classList.toggle('picked', !!rec && rec.label === 'preserved');
  $('no').classList.toggle('picked', !!rec && rec.label === 'not_preserved');
  $('comment').value = rec ? (rec.comment || '') : '';
}

async function submit(label) {
  const c = S.cases[S.idx];
  const comment = $('comment').value.trim();
  try {
    await api('/api/label', {annotator: S.annotator, case_id: c.case_id, label, comment});
  } catch (e) { alert('Save failed: ' + e.message); return; }
  S.labels[c.case_id] = {label, comment};
  S.showDone = true;
  const nxt = firstUnlabeled(S.idx + 1);
  if (nxt !== -1) S.idx = nxt;
  render();
}

$('yes').onclick = () => submit('preserved');
$('no').onclick = () => submit('not_preserved');
$('prev').onclick = () => { if (S.idx > 0) { S.idx--; S.showDone = false; render(); } };
$('next').onclick = () => { if (S.idx < S.cases.length - 1) { S.idx++; S.showDone = false; render(); } };
$('jump').onclick = () => { const i = firstUnlabeled(S.idx + 1); if (i !== -1) { S.idx = i; render(); } };
$('review').onclick = () => { S.idx = 0; S.showDone = false; render(); };

document.addEventListener('keydown', e => {
  if (!S || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  if (e.key === '1') $('yes').click();
  else if (e.key === '2') $('no').click();
  else if (e.key === 'ArrowLeft') $('prev').click();
  else if (e.key === 'ArrowRight') $('next').click();
});
</script>
</body>
</html>
"""


def main() -> None:
    global STUDY_DIR, ANN_DIR, CASES, CASE_IDS, MODE
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', default='./study', help='dir with cases.jsonl + images/')
    ap.add_argument('--annotations', default='./annotations')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--adjudicate', nargs=2, metavar=('ANN1', 'ANN2'),
                    help='serve only cases where these two primary annotators disagree')
    args = ap.parse_args()

    STUDY_DIR = Path(args.study)
    ANN_DIR = Path(args.annotations)
    CASES = load_cases(STUDY_DIR / 'cases.jsonl')
    CASE_IDS = sorted(CASES)

    if args.adjudicate:
        MODE = 'adjudicate'
        CASE_IDS = disagreement_ids(*args.adjudicate)
        print(f'Adjudication mode: {len(CASE_IDS)} disagreement case(s) between '
              f'{args.adjudicate[0]} and {args.adjudicate[1]}.')
        if not CASE_IDS:
            raise SystemExit('No disagreements — adjudication not needed.')
    else:
        print(f'Primary mode: {len(CASE_IDS)} cases.')

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f'Serving on http://{args.host}:{args.port}  (Ctrl-C to stop)')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
