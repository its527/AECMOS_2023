# AEC Blind-Test Re-Segmentation

Re-segment the [Microsoft AEC-Challenge ICASSP 2023 blind test set](https://github.com/microsoft/AEC-Challenge/tree/main/datasets/blind_test_set_icassp2023)
into **near-end-singletalk**, **far-end-singletalk**, and **doubletalk** clips by
*actual per-timestamp signal activity* instead of the original folder labels.

Each recording is a pair of WAV files — a microphone capture (`*_mic.wav`) and the
loudspeaker reference / loopback (`*_lpb.wav`). `segment_aec_pairs.py` scans the
reference and microphone frame by frame, decides which of the three scenarios is active
at each moment, and writes every contiguous stretch of ≥ 10 s as a new co-temporal
`(mic, lpb)` pair in the matching scenario folder.

## Contents

- **`segment_aec_pairs.py`** — the re-segmentation tool (main script)
- **`verify_pairs.py`** — optional QA pass that verifies far-end vs doubletalk on the resulting pairs

## Requirements

- Python 3.8+
- `numpy`, `scipy`, `soundfile`

```bash
pip install numpy scipy soundfile
```

## Usage

```bash
python segment_aec_pairs.py \
    --input  /path/to/blind_test_set_icassp2023 \
    --output /path/to/resegmented
```

`--input` is searched **recursively** for `*_mic.wav`; each is paired with the
`*_lpb.wav` beside it. Any sample rate / channel count is accepted — analysis runs on a
mono downmix, the reference is resampled to the mic rate for output, and everything is
written as 16-bit PCM.

### Input layout

```
blind_test_set_icassp2023/
├── doubletalk/
│   ├── <id>_mic.wav
│   ├── <id>_lpb.wav
│   └── ...
├── farend-singletalk/
└── nearend-singletalk/
```

The original folder names are **not** trusted — every pair is re-classified from its audio.

### Output layout

```
resegmented/
├── nearend-singletalk/
│   ├── <id>_nearend-singletalk_000_0012.3-0025.7s_mic.wav
│   ├── <id>_nearend-singletalk_000_0012.3-0025.7s_lpb.wav
│   └── ...
├── farend-singletalk/
├── doubletalk/
└── segments_manifest.json
```

Each output filename encodes the source id, the assigned scenario, a per-source index,
and the `start-end` time span (in seconds) taken from the original recording. The
written audio is the **original samples sliced for that span** — timing is never
modified. `segments_manifest.json` lists every segment with its source paths, scenario,
timing, duration, echo return loss, and output paths.

## How the classification works

For each pair the script:

1. **Aligns** the reference to the echo in the mic with a single GCC-PHAT delay estimate
   (used only internally, for the tests below — the written audio is never time-shifted).
2. **Far-end activity** — flags frames where the reference is playing (its level rises
   above an adaptive floor).
3. **Near-end activity** — a frame carries near-end speech when *both*:
   - the residual after cancelling the reference echo is **harmonic** (voiced speech is
     harmonic; broadband noise such as a fan, keyboard, or clatter is not), and
   - during far-end activity, the mic and reference are **incoherent** (leftover echo
     stays coherent with the reference; an independent near-end talker does not).
4. **Labels** each frame near-end-singletalk / far-end-singletalk / doubletalk / silence,
   smooths the label track, merges short runs, and writes every run ≥ `--min-seconds`.

## Options

**Required**

| flag | description |
|---|---|
| `--input`  | dataset root (searched recursively for `*_mic.wav`) |
| `--output` | output root; the three scenario folders are created here |

**Common**

| flag | default | description |
|---|---|---|
| `--min-seconds` | `10`   | minimum segment length to keep (s) |
| `--far-margin`  | `12`   | dB above the reference floor counted as far-end active |
| `--harm-thresh` | `0.40` | harmonicity needed to call a residual "speech" |
| `--coh-thresh`  | `0.70` | below this mic/reference coherence, far-end energy is treated as near-end |

Additional tuning flags (usually fine at their defaults): `--win`, `--hop`,
`--lpb-abs-floor`, `--echo-oversub`, `--near-resid-margin`, `--pitch-min`,
`--pitch-max`, `--dt-smooth`, `--far-hang`, `--near-hang`, `--max-delay`,
`--min-smooth`. Run `python segment_aec_pairs.py --help` for the full list.

## Verifying far-end vs doubletalk (optional)

Telling far-end from doubletalk is the hard part: when the echo is weak, reverberant, or
nonlinear it can look like a near-end talker, so the `doubletalk/` folder may pick up
some genuinely far-end pairs. `verify_pairs.py` re-checks each pair by asking whether the
mic carries independent energy during the reference's **silent** stretches (near-end →
doubletalk; mic falls to its floor → echo only), and can collect just the pairs it can
positively confirm:

```bash
# audit: verdict table + per-pair diagnostic plots
python verify_pairs.py --input resegmented/doubletalk --output verify_out

# extract only the high-confidence doubletalk pairs into a clean folder
python verify_pairs.py --input resegmented/doubletalk --collect clean_doubletalk --no-plots
```

Only **doubletalk** is ever reported as *confident* (a near-end talker positively caught
in a reference gap); far-end and ambiguous pairs are flagged for review rather than
guessed. See the header of `verify_pairs.py` for the precision/recall knobs.

## Known limitations

- The near-end test keys on **voiced** speech; a near-end talker that is almost entirely
  unvoiced may be under-detected.
- Far-end vs doubletalk cannot be resolved reliably for *every* pair from audio alone
  (weak / reverberant / nonlinear echo is genuinely ambiguous). Treat the `doubletalk/`
  split as a first pass and use `verify_pairs.py` when precision matters.
- Defaults are tuned for the ICASSP 2023 blind test set (the real files are 48 kHz mono).

## License

TODO — add a license before publishing (e.g. MIT).
