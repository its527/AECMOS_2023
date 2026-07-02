#!/usr/bin/env python3
"""
segment_aec_pairs.py

Re-segment AEC-Challenge blind-test pairs (<name>_mic.wav / <name>_lpb.wav) into
near-end singletalk, far-end singletalk, and double-talk regions based on the
ACTUAL signal activity, ignoring the original (unreliable) folder names.

Signal model
------------
    mic = near-end speech + echo + noise
    lpb = far-end reference  (the signal played out of the loudspeaker)
where echo is a DELAYED, ATTENUATED copy of lpb.

Scenario definitions
---------------------
    near-end singletalk : near-end active, far-end (lpb) inactive
    far-end  singletalk : far-end active, echo in mic, NO near-end
    double-talk         : near-end AND far-end active at the same time

How near-end is detected during far-end playback
------------------------------------------------
Echo also adds energy (and, since echo is just delayed far-end speech, harmonic
structure) to the mic, so neither energy nor "is there voice" can tell near-end
from echo on its own. We:
  1. Estimate the mic<->lpb bulk delay with GCC-PHAT and align lpb to the echo.
  2. Remove the echo: a minimum-residual (Wiener) per-band magnitude gain maps the
     reference to its echo and is spectral-subtracted from the mic.
  3. On the residual, require BOTH (a) harmonicity -- voiced speech is periodic,
     broadband/coloured noise is not, so a fan/keyboard/clatter burst is rejected;
     and (b) while the far-end plays, LOW mic/reference coherence -- leftover echo
     stays coherent with the reference, a near-end talker does not, so a strong
     harmonic echo cannot masquerade as near-end.
A frame is near-end when speech is present and (the far-end is silent OR the
coherence test says the energy is not echo).

Only contiguous segments >= --min-seconds (default 10s) are written out. Each
segment is saved as a co-temporal (mic, lpb) pair so it stays usable as an AEC
test case: the original mic and reference samples are sliced for the segment's time
span and written unmodified (the reference is only resampled if its rate differs from
the mic). Detection uses a single file-level alignment of the reference to the echo
internally; the written audio is not time-shifted.

Usage
-----
    pip install numpy scipy soundfile
    python segment_aec_pairs.py --input  /path/to/blind_test_set_icassp2023 \
                                --output /path/to/resegmented

All thresholds are exposed as CLI flags; defaults are tuned for 16 kHz AEC data.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency. Run:  pip install numpy scipy soundfile")

from scipy.signal import resample_poly
from scipy.ndimage import uniform_filter1d

# Class ids
SIL, NE, FE, DT = 0, 1, 2, 3
CLASS_DIR = {NE: "nearend-singletalk", FE: "farend-singletalk", DT: "doubletalk"}
CLASS_NAME = {SIL: "silence", NE: "nearend-singletalk", FE: "farend-singletalk", DT: "doubletalk"}


# ----------------------------------------------------------------------------- helpers
def gcc_phat(sig, ref, fs, max_tau):
    """Return integer delay (samples) such that sig[n] ~ ref[n - delay]."""
    n = len(sig) + len(ref)
    nfft = 1 << int(np.ceil(np.log2(n)))
    SIG = np.fft.rfft(sig, n=nfft)
    REF = np.fft.rfft(ref, n=nfft)
    R = SIG * np.conj(REF)
    R /= np.abs(R) + 1e-12  # PHAT weighting
    cc = np.fft.irfft(R, n=nfft)
    max_shift = min(int(fs * max_tau), nfft // 2)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    return int(np.argmax(np.abs(cc)) - max_shift)


def estimate_delay(mic, lpb, fs, max_delay):
    """Estimate echo delay on the loudest ~5 s window of lpb. Clamp to >= 0."""
    win = min(len(lpb), int(5 * fs))
    if win >= fs:  # find a high-energy excerpt so the echo is actually present
        env = np.convolve(lpb ** 2, np.ones(int(0.5 * fs)), "same")
        c = int(np.argmax(env))
        s = max(0, min(c - win // 2, len(lpb) - win))
        mic, lpb = mic[s : s + win], lpb[s : s + win]
    d = gcc_phat(mic, lpb, fs, max_delay)
    return max(d, 0)  # echo cannot precede the reference


def shift_signal(x, d):
    """Delay x by d samples along axis 0 (d>0 shifts later, zero-padded).

    Works for both 1-D (mono) and 2-D (samples, channels) arrays.
    """
    if d <= 0:
        return x
    if d >= len(x):
        return np.zeros_like(x)
    pad = np.zeros((d,) + x.shape[1:], dtype=x.dtype)
    return np.concatenate((pad, x[:-d]), axis=0)


def dilate(mask, k):
    """Symmetric binary dilation by k frames (hangover for tails / onsets)."""
    if k <= 0:
        return mask
    return np.convolve(mask.astype(int), np.ones(2 * k + 1), "same") > 0


def runs_of(label):
    """Run-length encode -> list of (start, end_exclusive, label)."""
    runs, start = [], 0
    for i in range(1, len(label)):
        if label[i] != label[i - 1]:
            runs.append((start, i, int(label[i - 1])))
            start = i
    runs.append((start, len(label), int(label[-1])))
    return runs


def smooth_runs(label, min_len):
    """Merge runs shorter than min_len frames into their longer neighbor.

    Prevents a brief spurious flip from chopping an otherwise long, valid segment.
    """
    label = label.copy()
    while True:
        runs = runs_of(label)
        if len(runs) == 1:
            break
        shorts = [r for r in runs if (r[1] - r[0]) < min_len]
        if not shorts:
            break
        r = min(shorts, key=lambda r: r[1] - r[0])
        i = runs.index(r)
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i < len(runs) - 1 else None
        if left and right:
            rep = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
        else:
            rep = (left or right)[2]
        label[r[0] : r[1]] = rep
    return label


# ----------------------------------------------------------------------------- core
def _stft(x, nfft, hop, window):
    """Short-time FFT -> complex array [frames, freqs]."""
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    frames = sliding_window_view(x, nfft)[::hop]
    return np.fft.rfft(frames.astype(np.float64) * window, axis=1)


def _smooth_time(X, L):
    """Moving average over frames (axis 0); handles complex input."""
    if np.iscomplexobj(X):
        return (uniform_filter1d(X.real, L, axis=0, mode="nearest")
                + 1j * uniform_filter1d(X.imag, L, axis=0, mode="nearest"))
    return uniform_filter1d(X, L, axis=0, mode="nearest")


def classify_frames(mic_m, lpb_al, fs, a):
    """Per-frame scenario labels.

    Near-end is detected on an ECHO-REMOVED residual with a voice-activity gate,
    so neither the far-end echo nor non-speech background noise is mistaken for a
    near-end talker.
    """
    # ~32 ms analysis window (power of two), scaled to the sample rate. A fixed
    # 512-pt window is ~32 ms at 16 kHz but only ~11 ms at 48 kHz, which collapses
    # frequency resolution and deflates the echo coherence -> far-end leaks into
    # double-talk. Scaling keeps behaviour consistent across 16 / 44.1 / 48 kHz
    # (and evaluates to exactly 512 at 16 kHz).
    nfft = 1 << int(np.ceil(np.log2(0.032 * fs)))
    hop = max(1, int(a.hop * fs))
    window = np.hanning(nfft)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    Mic = _stft(mic_m, nfft, hop, window)
    Lpb = _stft(lpb_al, nfft, hop, window)
    n = min(len(Mic), len(Lpb))
    Mic, Lpb = Mic[:n], Lpb[:n]
    Mmag, Lmag = np.abs(Mic), np.abs(Lpb)
    Pmm, Pll = Mmag ** 2, Lmag ** 2

    # ---- far-end activity from per-frame MEAN-SQUARE level (true dBFS scale, so
    # the absolute floor cap is meaningful). Summing |STFT|^2 would shift the scale
    # and the cap would land below the silence floor -> everything reads active.
    lpb_fr = sliding_window_view(lpb_al, nfft)[::hop][:n].astype(np.float64)
    lpb_db = 10 * np.log10((lpb_fr ** 2).mean(1) + 1e-12)
    lpb_floor = float(np.clip(min(np.percentile(lpb_db, 10), a.lpb_abs_floor),
                              -90.0, a.lpb_abs_floor))
    far_thr = lpb_floor + a.far_margin
    far_core = lpb_db > far_thr

    # ---- remove the echo: minimum-residual (Wiener) per-band magnitude gain |H(f)|
    # estimated on far-end frames, then spectral-subtract (with over-subtraction). A
    # Wiener gain (not a low percentile) returns ~1.0 when the echo is as loud as the
    # reference, so a strong echo cancels instead of leaking through as residual.
    if far_core.sum() >= 5:
        Mf, Lf = Mmag[far_core], Lmag[far_core]
        H = (Mf * Lf).sum(0) / ((Lf ** 2).sum(0) + 1e-12)
        H = np.clip(H, 0.0, 4.0)
    else:
        H = np.zeros(Pmm.shape[1])
    res_mag = np.maximum(Mmag - a.echo_oversub * H[None, :] * Lmag, 0.0)

    # ---- speech vs noise via HARMONICITY of the residual. Voiced speech is periodic
    # (sharp autocorrelation peak at the pitch lag); background noise -- broadband OR
    # coloured -- is not. Autocorrelation = inverse-FFT of the power spectrum (no phase
    # needed), so this is robust to noise colour unlike a spectral-flatness test.
    band = (freqs >= 200) & (freqs <= 3500)
    res_db = 10 * np.log10((res_mag[:, band] ** 2).sum(1) + 1e-12)
    ac = np.fft.irfft(res_mag ** 2, n=nfft, axis=1)
    lo = max(1, int(fs / a.pitch_max))
    hi = min(nfft // 2, int(fs / a.pitch_min))
    harm = ac[:, lo:hi].max(1) / (ac[:, 0] + 1e-12)

    # Residual noise floor from far-INACTIVE frames: the cancelled echo in far-active
    # frames drives res_db to -inf and would drag a global percentile far too low.
    quiet = res_db[~far_core] if (~far_core).sum() >= 20 else res_db
    res_floor = np.percentile(quiet, 10)
    speech = (harm > a.harm_thresh) & (res_db > res_floor + a.near_resid_margin)

    far_active = dilate(far_core, max(0, round(a.far_hang / a.hop)))

    # ---- echo vs near-end via magnitude-squared COHERENCE of mic with the aligned
    # reference. Leftover residual echo stays COHERENT with the reference (MSC ~ 1);
    # a near-end talker is uncorrelated with it (MSC < 1). This is the cue a strong,
    # harmonic echo cannot fake, so it -- not the residual level -- decides near-end
    # while the far-end is playing. (Where the far-end is silent there is no echo to
    # confuse, so the harmonic-speech test stands alone.)
    Sxy = _smooth_time(Mic * np.conj(Lpb), 8)
    Sxx = _smooth_time(Pmm, 8)
    Syy = _smooth_time(Pll, 8)
    msc = (np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-20))[:, band].mean(1)
    near_present = np.where(far_active, speech & (msc < a.coh_thresh), speech)

    # ---- smooth the near-end decision, then hangover its edges
    ks = max(1, round(a.dt_smooth / a.hop))
    near_present = np.convolve(near_present.astype(float), np.ones(ks) / ks, "same") > 0.5
    near_present = dilate(near_present, max(0, round(a.near_hang / a.hop)))

    label = np.full(n, SIL, dtype=int)
    label[near_present & ~far_active] = NE
    label[~near_present & far_active] = FE
    label[near_present & far_active] = DT
    label = smooth_runs(label, max(1, round(a.min_smooth / a.hop)))

    erl = 10 * np.log10(np.median(H[H > 0]) ** 2) if np.any(H > 0) else None
    return label, {"far_thr_db": round(far_thr, 1),
                   "erl_db": round(erl, 1) if erl is not None else None}


def process_pair(mic_path, lpb_path, a, manifest):
    mic, sr = sf.read(mic_path, always_2d=True)
    lpb, sr2 = sf.read(lpb_path, always_2d=True)
    if sr2 != sr:
        lpb = resample_poly(lpb, sr, sr2, axis=0)

    mic_m, lpb_m = mic.mean(axis=1), lpb.mean(axis=1)
    L = min(len(mic_m), len(lpb_m))
    mic_m, lpb_m = mic_m[:L], lpb_m[:L]
    L_full = min(len(mic), len(lpb))

    if L_full < a.min_seconds * sr:
        print(f"  skip (file shorter than {a.min_seconds}s)")
        return

    delay = estimate_delay(mic_m, lpb_m, sr, a.max_delay)
    lpb_al = shift_signal(lpb_m, delay)  # align reference with the echo in mic

    label, params = classify_frames(mic_m, lpb_al, sr, a)
    hop = int(a.hop * sr)
    win = int(a.win * sr)

    base = os.path.basename(mic_path)[: -len("_mic.wav")]
    erl_str = f"{params['erl_db']}dB" if params["erl_db"] is not None else "n/a"
    print(f"  ERL={erl_str}", end="")
    far_thr_db = params["far_thr_db"]

    kept, dropped = 0, 0
    for k, (fs_, fe_, lab) in enumerate(runs_of(label)):
        if lab == SIL:
            continue
        s = fs_ * hop
        e = min((fe_ - 1) * hop + win, L_full)
        if (e - s) < a.min_seconds * sr:
            continue

        # Backstop: a near-end-singletalk segment must have a (near-)silent
        # reference. If lpb is actually active over the segment, this is not
        # near-end singletalk -> drop it rather than mislabel it.
        if lab == NE:
            seg_lpb_db = 10 * np.log10(np.mean(lpb[s:e].mean(axis=1) ** 2) + 1e-12)
            if seg_lpb_db > far_thr_db:
                dropped += 1
                continue

        # Slice the ORIGINAL mic and reference for this segment and write them
        # unmodified (lpb was only resampled to the mic rate; timing is untouched).
        mic_seg = mic[s:e]
        lpb_seg = lpb[s:e]
        t0, t1 = s / sr, e / sr
        out_dir = os.path.join(a.output, CLASS_DIR[lab])
        stem = f"{base}_{CLASS_DIR[lab]}_{kept:03d}_{t0:06.1f}-{t1:06.1f}s"
        mic_out = os.path.join(out_dir, stem + "_mic.wav")
        lpb_out = os.path.join(out_dir, stem + "_lpb.wav")
        sf.write(mic_out, mic_seg, sr, subtype="PCM_16")
        sf.write(lpb_out, lpb_seg, sr, subtype="PCM_16")

        manifest.append({
            "source_mic": mic_path, "source_lpb": lpb_path,
            "scenario": CLASS_NAME[lab], "start_s": round(t0, 2), "end_s": round(t1, 2),
            "duration_s": round((e - s) / sr, 2),
            "erl_db": params["erl_db"], "mic_out": mic_out, "lpb_out": lpb_out,
        })
        kept += 1
    msg = f"  -> {kept} segment(s)"
    if dropped:
        msg += f"  ({dropped} near-end dropped: lpb active)"
    print(msg)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="dataset root (searched recursively for *_mic.wav)")
    p.add_argument("--output", required=True, help="output root for the 3 scenario folders")
    p.add_argument("--min-seconds", type=float, default=10.0,
                   help="minimum length of a kept segment (default 10)")
    # framing
    p.add_argument("--win", type=float, default=0.025, help="analysis window, s")
    p.add_argument("--hop", type=float, default=0.010, help="analysis hop, s")
    # far-end activity
    p.add_argument("--far-margin", type=float, default=12.0,
                   help="dB above lpb noise floor to call far-end active")
    p.add_argument("--lpb-abs-floor", type=float, default=-60.0,
                   help="absolute cap (dBFS) on the estimated lpb noise floor; stops a "
                        "continuously-active reference from inflating it (far-end files)")
    # near-end detection: echo removal + residual voice-activity gate
    p.add_argument("--echo-oversub", type=float, default=1.5,
                   help="over-subtraction factor when removing echo before the VAD "
                        "(higher = more echo removed, but suppresses weak near-end)")
    p.add_argument("--near-resid-margin", type=float, default=8.0,
                   help="dB the residual must exceed its floor to contain a near-end source")
    p.add_argument("--harm-thresh", type=float, default=0.40,
                   help="min residual harmonicity (0-1) to count as near-end SPEECH; "
                        "RAISE to reject more noise, LOWER to catch weaker/voiced-sparse speech")
    p.add_argument("--coh-thresh", type=float, default=0.70,
                   help="mic/reference coherence below which a far-end frame is judged to "
                        "contain a near-end source (above = pure echo). Real (nonlinear) "
                        "echo sits ~0.9; near-end speech pulls it well below. LOWER sends "
                        "more to far-end, HIGHER sends more to double-talk (default 0.70)")
    p.add_argument("--pitch-min", type=float, default=70.0,
                   help="lowest near-end pitch, Hz (harmonicity search)")
    p.add_argument("--pitch-max", type=float, default=400.0,
                   help="highest near-end pitch, Hz (harmonicity search)")
    p.add_argument("--dt-smooth", type=float, default=0.15,
                   help="smoothing window, s, for the near-end (double-talk) decision")
    p.add_argument("--far-hang", type=float, default=0.20,
                   help="far-end hangover, s, to cover the reverberant echo tail")
    p.add_argument("--near-hang", type=float, default=0.15,
                   help="near-end hangover, s, to smooth talk-spurt edges")
    p.add_argument("--max-delay", type=float, default=0.5, help="max mic/lpb delay to search, s")
    p.add_argument("--min-smooth", type=float, default=0.4,
                   help="runs shorter than this (s) are merged into neighbors")
    a = p.parse_args()

    for d in CLASS_DIR.values():
        os.makedirs(os.path.join(a.output, d), exist_ok=True)

    mic_files = sorted(glob.glob(os.path.join(a.input, "**", "*_mic.wav"), recursive=True))
    if not mic_files:
        sys.exit(f"No *_mic.wav files found under {a.input}")

    manifest = []
    for mic_path in mic_files:
        lpb_path = mic_path[: -len("_mic.wav")] + "_lpb.wav"
        print(f"[{os.path.relpath(mic_path, a.input)}]")
        if not os.path.exists(lpb_path):
            print("  skip (no matching _lpb.wav)")
            continue
        try:
            process_pair(mic_path, lpb_path, a, manifest)
        except Exception as exc:  # keep going on a bad file
            print(f"  ERROR: {exc}")

    with open(os.path.join(a.output, "segments_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    counts = {c: 0 for c in CLASS_NAME.values() if c != "silence"}
    for r in manifest:
        counts[r["scenario"]] += 1
    print("\n=== Summary ===")
    for name, n in counts.items():
        print(f"  {name:20s}: {n} segments")
    print(f"  manifest: {os.path.join(a.output, 'segments_manifest.json')}")


if __name__ == "__main__":
    main()
