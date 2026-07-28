# Image Annotation Study

Thank you for helping with this study! You will look at pairs of images and
judge whether a generated image still shows the **same class** as the original
source image.

Everything you need is already in this repository — all study images are
included, and the tool runs with **plain Python 3 only** (no packages to
install, no setup).

## Quick Start

1. **Clone this repository** (or download it as a ZIP and unpack it):

   ```bash
   git clone https://github.com/beanduan22/Imagelabel.git
   cd Imagelabel
   ```

2. **Warm up first** with 9 practice items (these are *not* counted in the
   study — use them to get familiar with the interface):

   ```bash
   python server.py --study ./demo --port 8765
   ```

   Open <http://localhost:8765> in your browser and enter your annotator ID
   (the ID given to you by the organizer, e.g. `alice`). Click through the
   practice items, then stop the server with `Ctrl+C`.

3. **Start the real session**:

   ```bash
   python server.py --study ./study --port 8765
   ```

   Open <http://localhost:8765> again and log in with the **same ID**.
   Work through all cases.

4. **When you are finished**, send the file `annotations/<yourID>.jsonl`
   back to the organizer.

## What You Will See and Judge

For each case, the interface shows two images side by side:

- the **source image**;
- the **generated image** derived from it.

Your task for every case:

> Does the generated image still depict the same class of object as the
> source image?

- Choose **Yes, preserved** if the generated image still shows an object of
  the same class as the source image.
- Choose **No, not preserved** if it no longer does (the object is
  unrecognizable, or it now looks like a different class).

Trust your own perception — there is no "trick"; just answer what you see.

## Interface Tips

- **Keyboard shortcuts:** `1` = Yes (preserved), `2` = No (not preserved),
  `←` / `→` = previous / next case.
- Every answer is **saved instantly** to `annotations/<yourID>.jsonl`.
  You can close the browser or stop the server at any time and resume later —
  your progress is kept.
- If you change your mind, just re-answer a case; the **last answer counts**.
- Please work **independently**: do not discuss cases or answers with other
  annotators while the study is running.

## For Study Organizers

- Each annotator sees the cases in an order deterministically shuffled by
  their ID (different order per annotator, stable across sessions).
- After both primary annotators finish, run the adjudication mode for the
  third annotator, who only sees the cases where the two disagree (without
  seeing their answers):

  ```bash
  python server.py --study ./study --port 8765 --adjudicate alice bob
  ```

- Compute the final statistics (proportions with 95% Wilson CIs, Cohen's κ,
  per-combination breakdown) and write `study/validation_summary.csv`:

  ```bash
  python analyze.py --study ./study --a1 alice --a2 bob --adj carol
  ```

## Files

| Path | Purpose |
|---|---|
| `server.py` | Annotation GUI server (standard library only) |
| `study/` | The study cases and images |
| `demo/` | 9 practice items (not part of the study) |
| `annotations/` | One `.jsonl` file per annotator, written automatically |
| `analyze.py` | Statistics script (organizer use) |
