#!/usr/bin/env python3
"""Verify each mic/lpb pair as FAR-END vs DOUBLETALK, the way the analysis was done by
hand -- with a decisive test plus the evidence and plots behind it.

The reliable question is NOT "how strong/coherent is the echo" (that varies wildly and
fooled every level/coherence heuristic). It is: **is there a near-end talker?** A
near-end talker is the ONLY thing that distinguishes doubletalk from far-end, and it
shows itself in one robust way -- energy in the mic that the reference cannot explain,
most cleanly during the reference's own SILENCES:

    reference silent  +  mic stays energized   -> near-end present  -> DOUBLETALK
    reference silent  +  mic falls to its floor -> only echo         -> FAR-END

Reverb tail briefly holds the mic up after the reference stops, so the test looks
*deep* into each silence (after a guard) rather than at its edge.

That test is decisive only when the reference actually goes silent long enough. When the
reference plays continuously (no gaps), near-end presence cannot be proven from gaps, and
-- as the analysis showed -- no single number (coherence, echo return loss, envelope
correlation, mic-drop) separates the two reliably. Those pairs are marked REVIEW: the
script reports the supporting evidence and writes a diagnostic plot (mic + reference
spectrograms, level envelopes with the silences shaded, and the mic-vs-reference tracking
curve) so the call can be made by eye in seconds -- exactly how it was done here.

Verdicts:
    farend / doubletalk   (confident)  -- from the reference-silence test
    farend? / doubletalk? (review)     -- continuous reference; evidence leans this way
    review                             -- continuous reference; evidence inconclusive

Only DOUBLETALK is ever "confident" (near-end positively caught in a reference gap), so
--collect keeps just those pairs: a clean, high-precision doubletalk set with no far-end
contamination. Recall is traded for precision on purpose -- doubletalk whose near-end
never lands in a reference gap is dropped rather than risk polluting the set.

Usage:
    python verify_pairs.py --input <folder>                        # table + plots into ./verify_out
    python verify_pairs.py --input <folder> --collect <clean_dt>   # copy ONLY confident doubletalk
    python verify_pairs.py --input <folder> --collect <clean_dt> --no-plots   # fast extraction

Recall/precision knobs (the silence test only fires when the reference actually pauses):
    --min-silence   (default 0.3 s) reference gap needed to run the test; lower => more pairs testable
    --silence-guard (default 0.15 s) ignored after the reference stops, for reverb tail
    --near-frac     (default 0.15)  fraction of a gap that must light up to call doubletalk
    --near-burst-db (default 10 dB) how far above the mic floor a frame must be to count as near-end
Lower --min-silence / --silence-guard for MORE confident doubletalk (phrase-level pauses); raise
--near-frac / --silence-guard for STRICTER precision. Pairs whose reference never pauses at all
cannot be verified by this test and stay "review" regardless of these knobs.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency. Run:  pip install numpy scipy soundfile matplotlib")

from scipy.signal import resample_poly
from scipy.ndimage import binary_erosion, uniform_filter1d

# ---- decision constant DEFAULTS (all overridable on the command line) ---------
MIN_SILENCE_S   = 0.3    # sustained reference silence needed to run the silence test
SILENCE_GUARD_S = 0.15   # skip this long after the reference stops (reverb tail)
NEAR_BURST_DB   = 10.0   # a silent frame with mic this far above its floor = near-end burst
NEAR_FRAC       = 0.15   # this fraction of silent frames bursting => near-end present
GAP_FAR_DB      = 6.0    # (continuous ref) mic within this of its floor at dips => lean far-end
GAP_DT_DB       = 12.0   # (continuous ref) mic this far above its floor at dips => lean doubletalk


# ----------------------------------------------------------------------------- DSP
def gcc_phat(sig, ref, fs, max_tau):
    n = len(sig) + len(ref)
    nfft = 1 << int(np.ceil(np.log2(n)))
    R = np.fft.rfft(sig, n=nfft) * np.conj(np.fft.rfft(ref, n=nfft))
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n=nfft)
    ms = min(int(fs * max_tau), nfft // 2)
    cc = np.concatenate((cc[-ms:], cc[: ms + 1]))
    return int(np.argmax(np.abs(cc)) - ms)


def estimate_delay(mic, lpb, fs, max_delay=0.5):
    win = min(len(lpb), int(5 * fs))
    if win >= fs:
        env = np.convolve(lpb ** 2, np.ones(int(0.5 * fs)), "same")
        c = int(np.argmax(env))
        s = max(0, min(c - win // 2, len(lpb) - win))
        mic, lpb = mic[s : s + win], lpb[s : s + win]
    return max(gcc_phat(mic, lpb, fs, max_delay), 0)


def shift_signal(x, d):
    if d <= 0:
        return x
    if d >= len(x):
        return np.zeros_like(x)
    return np.concatenate((np.zeros(d, dtype=x.dtype), x[:-d]))


def _stft_mag(x, nfft, hop, w, n):
    return np.abs(np.fft.rfft(sliding_window_view(x, nfft)[::hop][:n] * w, axis=1))


def _smooth(X, k):
    if np.iscomplexobj(X):
        return uniform_filter1d(X.real, k, axis=0) + 1j * uniform_filter1d(X.imag, k, axis=0)
    return uniform_filter1d(X, k, axis=0)


# ----------------------------------------------------------------------------- analysis
def analyze(mic_path, lpb_path, cfg):
    mic, sr = sf.read(mic_path, always_2d=True)
    lpb, sr2 = sf.read(lpb_path, always_2d=True)
    if sr2 != sr:
        lpb = resample_poly(lpb, sr, sr2, axis=0)
    mic_m, lpb_m = mic.mean(1), lpb.mean(1)
    L = min(len(mic_m), len(lpb_m))
    mic_m, lpb_m = mic_m[:L], lpb_m[:L]

    d = estimate_delay(mic_m, lpb_m, sr)
    lpb_al = shift_signal(lpb_m, d)

    nfft = 1 << int(np.ceil(np.log2(0.032 * sr)))
    hop = int(0.010 * sr)
    n = 1 + (L - nfft) // hop
    if n < 20:
        return None
    t = np.arange(n) * hop / sr

    mic_db = 10 * np.log10((sliding_window_view(mic_m, nfft)[::hop][:n].astype(np.float64) ** 2).mean(1) + 1e-12)
    ref_db = 10 * np.log10((sliding_window_view(lpb_al, nfft)[::hop][:n].astype(np.float64) ** 2).mean(1) + 1e-12)

    ref_floor = np.clip(min(np.percentile(ref_db, 10), -60), -90, -60)
    far = ref_db > ref_floor + 12
    mic_floor = float(np.percentile(mic_db, 5))

    # --- decisive test: near-end BURSTS deep inside sustained reference silence.
    # Near-end speech is intermittent, so measure the FRACTION of silent frames that
    # light up (not the median, which the talker's own pauses drag down).
    gap = ~far
    g = int(cfg.silence_guard / 0.010)
    sustained = binary_erosion(gap, structure=np.ones(2 * g + 1)) if gap.any() else np.zeros_like(gap)
    sil_s = float(sustained.sum() * 0.010)
    if sustained.sum() >= max(15, int(cfg.min_silence / 0.010)):
        above = mic_db[sustained] - mic_floor
        near_frac = float(np.mean(above > cfg.near_burst_db))
        near_p90 = float(np.percentile(above, 90))
    else:
        near_frac = near_p90 = None

    # --- supporting evidence for the continuous-reference case
    hi = ref_db >= np.percentile(ref_db, 80)
    lo = ref_db <= np.percentile(ref_db, 20)
    mic_drop = float(np.median(mic_db[hi]) - np.median(mic_db[lo]))
    dips = ref_db <= np.percentile(ref_db[far], 15) if far.sum() > 20 else lo
    gap_at_dips = float(np.median(mic_db[dips]) - mic_floor)   # how far mic sits above floor when ref is weakest

    # --- context features
    mrr = float(np.median(mic_db[far] - ref_db[far])) if far.sum() > 20 else None
    w = np.hanning(nfft)
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    band = (fr >= 200) & (fr <= 3500)
    fmic = np.fft.rfft(sliding_window_view(mic_m, nfft)[::hop][:n] * w, axis=1)
    flpb = np.fft.rfft(sliding_window_view(lpb_al, nfft)[::hop][:n] * w, axis=1)
    Sxy = _smooth(fmic * np.conj(flpb), 8)
    Sxx = _smooth(np.abs(fmic) ** 2, 8)
    Syy = _smooth(np.abs(flpb) ** 2, 8)
    msc = (np.abs(Sxy) ** 2 / (Sxx * Syy + 1e-20))[:, band].mean(1)
    coh = float(np.median(msc[far])) if far.sum() > 20 else None

    # --- verdict.  Only DOUBLETALK can be proven here (near-end caught in a gap).
    # Absence of near-end in the gaps is consistent with far-end but cannot exclude a
    # near-end talker that is active only WHILE the far-end plays, so far-end is never
    # "confident" -- it is "likely" (had silences to check) or "review" (continuous ref).
    if sil_s >= cfg.min_silence and near_frac is not None and near_frac >= cfg.near_frac:
        verdict, conf = "doubletalk", "confident"
        reason = (f"near-end speech present during {sil_s:.1f}s of reference silence "
                  f"({near_frac:.0%} of silent frames energized, up to {near_p90:+.0f} dB) -> doubletalk")
    elif sil_s >= cfg.min_silence:
        verdict, conf = "farend", "likely"
        reason = (f"no near-end found in {sil_s:.1f}s of reference silence (mic stays at its floor) "
                  f"-> consistent with echo only; a near-end active only during far-end speech can't be ruled out from gaps")
    elif gap_at_dips <= cfg.gap_far_db and mic_drop >= 12:
        verdict, conf = "farend", "review"
        reason = (f"reference never silent; mic tracks it down to within {gap_at_dips:.0f} dB of its floor "
                  f"(drop {mic_drop:.0f} dB) -> no independent near-end seen, but not confirmable from gaps")
    elif gap_at_dips >= cfg.gap_dt_db:
        verdict, conf = "doubletalk", "review"
        reason = (f"reference never silent; mic stays {gap_at_dips:.0f} dB above its floor even at reference dips "
                  f"-> possible persistent near-end")
    else:
        verdict, conf = "review", "review"
        reason = "reference never silent and evidence inconclusive -> inspect the plot"

    return {
        "sr": sr, "dur_s": round(L / sr, 1), "delay_ms": round(d / sr * 1000, 1),
        "verdict": verdict, "confidence": conf, "reason": reason,
        "far_active_pct": round(100 * far.mean(), 1),
        "sustained_silence_s": round(sil_s, 1),
        "near_burst_frac": None if near_frac is None else round(near_frac, 2),
        "near_p90_db": None if near_p90 is None else round(near_p90, 1),
        "mic_drop_db": round(mic_drop, 1), "gap_at_ref_dips_db": round(gap_at_dips, 1),
        "mrr_db": None if mrr is None else round(mrr, 1),
        "coherence": None if coh is None else round(coh, 2),
        "mic_rms_db": round(10 * np.log10(np.mean(mic_m ** 2) + 1e-12), 1),
        "ref_rms_db": round(10 * np.log10(np.mean(lpb_m ** 2) + 1e-12), 1),
        # arrays for plotting (not serialized)
        "_plot": dict(t=t, mic_db=mic_db, ref_db=ref_db, far=far, sustained=sustained,
                      mic_floor=mic_floor, mic_m=mic_m, lpb_al=lpb_al, sr=sr, nfft=nfft, hop=hop),
    }


def make_plot(base, r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = r["_plot"]
    sr, nfft, hop = P["sr"], P["nfft"], P["hop"]
    w = np.hanning(nfft)
    fr = np.fft.rfftfreq(nfft, 1 / sr)
    fm = fr <= 4000

    def spec(x):
        S = np.abs(np.fft.rfft(sliding_window_view(x, nfft)[::hop] * w, axis=1))
        return 20 * np.log10(S[:, fm].T + 1e-6)

    Sm, Sl = spec(P["mic_m"]), spec(P["lpb_al"])
    Nm = min(Sm.shape[1], len(P["t"]))
    t = P["t"][:Nm]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.1, 1.2], hspace=0.45)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0); ax3 = fig.add_subplot(gs[3])

    ax0.pcolormesh(t, fr[fm], Sm[:, :Nm], vmin=-70, vmax=-10, cmap="magma", shading="auto")
    ax0.set_title(f"{base}    verdict: {r['verdict']} ({r['confidence']})", fontsize=10, loc="left")
    ax0.set_ylabel("mic  Hz")
    ax1.pcolormesh(t, fr[fm], Sl[:, :Nm], vmin=-70, vmax=-10, cmap="magma", shading="auto")
    ax1.set_ylabel("ref  Hz")

    # level envelopes with sustained-silence shaded + mic floor line
    tt = P["t"]
    ax2.plot(tt, P["mic_db"], lw=0.8, label="mic")
    ax2.plot(tt, P["ref_db"], lw=0.8, alpha=0.7, label="reference")
    ax2.axhline(P["mic_floor"], color="k", ls=":", lw=0.8, label="mic floor")
    sus = P["sustained"]
    if sus.any():  # shade sustained reference-silence regions
        edges = np.flatnonzero(np.diff(np.r_[0, sus.astype(int), 0]))
        for a, b in edges.reshape(-1, 2):
            ax2.axvspan(tt[a], tt[min(b, len(tt) - 1)], color="tab:green", alpha=0.15)
        ax2.axvspan(np.nan, np.nan, color="tab:green", alpha=0.15, label="ref silence (checked)")
    ax2.set_ylabel("level dB"); ax2.set_xlabel("time (s)")
    ax2.legend(fontsize=7, ncol=4, loc="upper right"); ax2.set_xlim(tt[0], tt[-1])

    # mic-vs-reference tracking curve (binned)
    md, rd = P["mic_db"], P["ref_db"]
    edges = np.percentile(rd, np.linspace(0, 100, 11))
    cx, cy, err = [], [], []
    for i in range(len(edges) - 1):
        msk = (rd >= edges[i]) & (rd <= edges[i + 1])
        if msk.sum() >= 5:
            cx.append((edges[i] + edges[i + 1]) / 2); cy.append(np.median(md[msk]))
            err.append([np.median(md[msk]) - np.percentile(md[msk], 25),
                        np.percentile(md[msk], 75) - np.median(md[msk])])
    if cx:
        err = np.array(err).T
        ax3.errorbar(cx, cy, yerr=err, fmt="o-", ms=4, capsize=3)
    ax3.axhline(P["mic_floor"], color="k", ls=":", lw=0.8, label="mic floor")
    ax3.set_xlabel("reference level (dB)"); ax3.set_ylabel("mic level (dB)")
    ax3.set_title("mic vs reference (rising & reaching floor = echo; flat/elevated = near-end)", fontsize=9)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.text(0.98, 0.02, r["reason"], transform=ax3.transAxes, fontsize=7.5,
             ha="right", va="bottom", wrap=True, color="dimgray")

    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="folder searched recursively for *_mic.wav")
    p.add_argument("--output", default="verify_out", help="where to write the report + plots")
    p.add_argument("--collect", metavar="DIR",
                   help="copy ONLY the confident-doubletalk pairs into DIR (the clean, "
                        "high-precision set); far-end and review pairs are ignored")
    p.add_argument("--confident-plots-only", action="store_true",
                   help="only render plots for confident-doubletalk pairs (skip the ignored ones)")
    p.add_argument("--no-plots", action="store_true", help="skip the diagnostic plots entirely")
    # precision/recall knobs (defaults catch phrase-level pauses; raise for stricter precision)
    p.add_argument("--min-silence", type=float, default=MIN_SILENCE_S,
                   help=f"reference silence (s) required to run the near-end test (default {MIN_SILENCE_S})")
    p.add_argument("--silence-guard", type=float, default=SILENCE_GUARD_S,
                   help=f"ignore this long (s) after the reference stops, for reverb tail (default {SILENCE_GUARD_S})")
    p.add_argument("--near-frac", type=float, default=NEAR_FRAC,
                   help=f"fraction of silent frames that must light up to call doubletalk (default {NEAR_FRAC})")
    p.add_argument("--near-burst-db", type=float, default=NEAR_BURST_DB,
                   help=f"dB above the mic floor for a silent frame to count as near-end (default {NEAR_BURST_DB})")
    p.add_argument("--gap-far-db", type=float, default=GAP_FAR_DB, help=argparse.SUPPRESS)
    p.add_argument("--gap-dt-db", type=float, default=GAP_DT_DB, help=argparse.SUPPRESS)
    a = p.parse_args()

    mics = sorted(glob.glob(os.path.join(a.input, "**", "*_mic.wav"), recursive=True))
    if not mics:
        sys.exit(f"No *_mic.wav found under {a.input}")
    os.makedirs(a.output, exist_ok=True)
    plot_dir = os.path.join(a.output, "plots")
    if not a.no_plots:
        os.makedirs(plot_dir, exist_ok=True)
    if a.collect:
        os.makedirs(a.collect, exist_ok=True)

    rows, report, collected = [], [], 0
    for mic_path in mics:
        lpb_path = mic_path[: -len("_mic.wav")] + "_lpb.wav"
        base = os.path.basename(mic_path)[: -len("_mic.wav")]
        if not os.path.exists(lpb_path):
            print(f"  [skip] no reference for {base}"); continue
        try:
            r = analyze(mic_path, lpb_path, a)
        except Exception as e:
            print(f"  [skip] {base}: {e}"); continue
        if r is None:
            print(f"  [skip] {base}: too short"); continue

        is_confident_dt = (r["verdict"] == "doubletalk" and r["confidence"] == "confident")

        if a.collect and is_confident_dt:
            shutil.copy2(mic_path, os.path.join(a.collect, os.path.basename(mic_path)))
            shutil.copy2(lpb_path, os.path.join(a.collect, os.path.basename(lpb_path)))
            collected += 1

        plot_rel = ""
        want_plot = not a.no_plots and (is_confident_dt or not a.confident_plots_only)
        if want_plot:
            plot_rel = os.path.join("plots", base + ".png")
            try:
                make_plot(base, r, os.path.join(a.output, plot_rel))
            except Exception as e:
                print(f"  [warn] plot failed for {base}: {e}"); plot_rel = ""

        rows.append((r["verdict"], r["confidence"], r["near_burst_frac"],
                     r["gap_at_ref_dips_db"], r["coherence"], r["mrr_db"], base))
        rec = {k: v for k, v in r.items() if k != "_plot"}
        rec.update(source_mic=mic_path, source_lpb=lpb_path, plot=plot_rel)
        report.append(rec)

    order = {"confident": 0, "likely": 1, "review": 2}
    rows.sort(key=lambda x: (order.get(x[1], 3), x[0]))
    print(f"\n{'verdict':<11} {'conf':<10} {'burst%':>7} {'dip dB':>7} {'coh':>5} {'MRR':>6}  file")
    print("-" * 92)
    for v, c, nf, gd, coh, mrr, base in rows:
        burst = f"{nf*100:5.0f}%" if nf is not None else "  n/a"
        f = lambda x, w, p="+.0f": (("%"+p) % x).rjust(w) if x is not None else "n/a".rjust(w)
        print(f"{v:<11} {c:<10} {burst:>7} {f(gd,7)} {f(coh,5,'.2f')} {f(mrr,6)}  {base}")
    print("-" * 92)
    n_conf = sum(1 for r in report if r["confidence"] == "confident")
    likely_far = sum(1 for r in report if r["confidence"] == "likely")
    rev = [r for r in report if r["confidence"] == "review"]
    rev_dt = sum(1 for r in rev if r["verdict"] == "doubletalk")
    rev_far = sum(1 for r in rev if r["verdict"] == "farend")
    rev_inc = sum(1 for r in rev if r["verdict"] == "review")
    print(f"{len(report)} pairs: {n_conf} confident doubletalk | {likely_far} likely far-end | "
          f"{len(rev)} review (of which ~{rev_dt} lean doubletalk, {rev_far} lean far-end, {rev_inc} inconclusive)"
          + ("" if a.no_plots else f"   (plots in {plot_dir}/)"))
    if a.collect:
        print(f"Collected {collected} confident-doubletalk pair(s) into {a.collect}/  "
              f"(far-end + review ignored)")

    with open(os.path.join(a.output, "verify_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report: {os.path.join(a.output, 'verify_report.json')}")


if __name__ == "__main__":
    main()
