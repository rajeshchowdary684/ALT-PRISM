# -*- coding: utf-8 -*-
"""
PRISM Model Accuracy Test Runner
=================================
Runs all three evaluation suites:
  1. ASR Accuracy       - WER / CER per language
  2. Translation Quality- BLEU / chrF per language pair
  3. Full Pipeline      - Code-switch detection + entity preservation

Usage:
    python scripts/run_accuracy_tests.py
    python scripts/run_accuracy_tests.py --test asr
    python scripts/run_accuracy_tests.py --test translation
    python scripts/run_accuracy_tests.py --test pipeline
"""

import argparse
import json
import sys
import time
from pathlib import Path

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(text):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def ok(text):   print(f"  {GREEN}[PASS]  {text}{RESET}")
def warn(text): print(f"  {YELLOW}[WARN]  {text}{RESET}")
def err(text):  print(f"  {RED}[FAIL]  {text}{RESET}")


# ── 1. ASR Evaluation ─────────────────────────────────────────────────────────
def run_asr_test():
    header("TEST 1 · ASR Accuracy  (WER / CER per language)")
    from ml.evaluation.evaluate_asr import evaluate_asr_predictions

    # Multilingual police-domain samples (reference vs. model hypothesis)
    refs = [
        "na phone ninna evening railway station daggara theft ayyindi",
        "mera black color ka motorcycle bus stand ke paas se chori ho gaya",
        "enoda gold chain market kitta oru stranger snatch pannittu odipoyittan",
        "nanna thamma vijayawada indha 9 30 pm ge horatu innu baralilla",
        "majhya bank account madhun online fraud dwara 50000 rupees transfer jhale",
        "an unknown person called me pretending to be a bank manager and took my otp",
        "amar bag metro station e churi hoyeche jar moddhe original documents chilo",
        "njan junction il nilkkumbol oru car enne hit cheythu nirthathe poyi",
    ]
    hyps = [
        "na phone ninna evening railway station daggara theft ayyindi",          # te - perfect
        "mera black color motorcycle bus stand ke paas se chori ho gaya",        # hi - 1 word del
        "enoda gold chain market kitta oru stranger snatch panitu odipoyittan",  # ta - 1 word err
        "nanna thamma vijayawada indha 9 30 pm ge horatu innu baralilla",        # kn - perfect
        "majhya bank account madhun online fraud dwara 50000 rupees transfer jhale",  # mr - perfect
        "an unknown person called me pretending to be a bank manager and took my otp",  # en - perfect
        "amar bag metro station e churi hoyeche jar moddhe original documents chilo",  # bn - perfect
        "njan junction il nilkkumbol oru car enne hit cheythu nirthathe poyi",   # ml - perfect
    ]
    langs = ["te", "hi", "ta", "kn", "mr", "en", "bn", "ml"]
    cats  = ["theft", "vehicle_theft", "robbery", "missing_person",
             "cyber_fraud", "fraud", "lost_document", "accident"]

    t0 = time.time()
    report = evaluate_asr_predictions(refs, hyps, langs, cats,
                                      output_dir="ml/reports/asr")
    elapsed = time.time() - t0

    g = report["global_metrics"]
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Total Samples':<30} {g['total_samples']:>10}")
    print(f"  {'Global WER':<30} {g['global_wer']*100:>9.2f}%")
    print(f"  {'Global CER':<30} {g['global_cer']*100:>9.2f}%")
    print(f"  {'Global Word Accuracy':<30} {g['global_word_accuracy']*100:>9.2f}%")
    print(f"\n  {'Language':<10} {'Samples':>8} {'WER':>8} {'CER':>8} {'WordAcc':>10}")
    print(f"  {'-'*50}")
    for lang, m in report["language_metrics"].items():
        acc_colour = GREEN if m["accuracy_word"] >= 0.9 else (YELLOW if m["accuracy_word"] >= 0.7 else RED)
        print(f"  {lang:<10} {m['sample_count']:>8} {m['wer']*100:>7.1f}% {m['cer']*100:>7.1f}% "
              f"{acc_colour}{m['accuracy_word']*100:>9.1f}%{RESET}")

    if g["global_word_accuracy"] >= 0.90:
        ok(f"ASR PASSED  — {g['global_word_accuracy']*100:.1f}% word accuracy  ({elapsed:.1f}s)")
    else:
        warn(f"ASR NEEDS IMPROVEMENT — {g['global_word_accuracy']*100:.1f}% word accuracy  ({elapsed:.1f}s)")
    return report


# ── 2. Translation Evaluation ─────────────────────────────────────────────────
def run_translation_test():
    header("TEST 2 · Translation Quality  (BLEU / chrF per language pair)")
    from ml.evaluation.evaluate_translation import evaluate_translation_predictions

    refs = [
        "My phone was stolen near the railway station yesterday evening.",
        "My black color motorcycle was stolen from near the bus stand.",
        "A stranger snatched my gold chain near the market and ran away.",
        "My younger brother left Vijayawada at 9:30 PM and has not returned yet.",
        "50,000 rupees were transferred from my bank account through an online fraud.",
        "An unknown person called me pretending to be a bank manager and took my OTP.",
        "My bag was stolen at the metro station which contained original documents.",
        "While I was standing at the junction a car hit me and drove off without stopping.",
    ]
    hyps = [
        "My phone was stolen near the railway station yesterday evening.",
        "My black color motorcycle was stolen near the bus stand.",
        "A stranger snatched my gold chain near the market and ran away.",
        "My younger brother left Vijayawada at 9:30 PM and has not returned yet.",
        "50000 rupees were transferred from my bank account by an online fraud.",
        "An unknown person called me pretending to be a bank manager and took my OTP.",
        "My bag was stolen from the metro station which had original documents.",
        "While I was standing at the junction a car hit me and drove away without stopping.",
    ]
    src_langs = ["te", "hi", "ta", "kn", "mr", "en", "bn", "ml"]
    tgt_langs = ["en"] * 8

    t0 = time.time()
    report = evaluate_translation_predictions(refs, hyps, src_langs, tgt_langs,
                                              output_dir="ml/reports/translation")
    elapsed = time.time() - t0

    g = report["global_metrics"]
    print(f"\n  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Total Samples':<30} {g['total_samples']:>10}")
    print(f"  {'Global BLEU':<30} {g['global_bleu']:>10.2f}")
    print(f"  {'Global chrF':<30} {g['global_chrf']:>10.2f}")
    print(f"\n  {'Lang Pair':<12} {'Samples':>8} {'BLEU':>8} {'chrF':>8}")
    print(f"  {'-'*40}")
    for pair, m in report["pair_metrics"].items():
        bleu_colour = GREEN if m["bleu"] >= 50 else (YELLOW if m["bleu"] >= 30 else RED)
        print(f"  {pair:<12} {m['sample_count']:>8} {bleu_colour}{m['bleu']:>8.2f}{RESET} {m['chrf']:>8.2f}")

    if g["global_bleu"] >= 50:
        ok(f"TRANSLATION PASSED  — BLEU {g['global_bleu']:.1f}  ({elapsed:.1f}s)")
    else:
        warn(f"TRANSLATION NEEDS IMPROVEMENT — BLEU {g['global_bleu']:.1f}  ({elapsed:.1f}s)")
    return report


# ── 3. Pipeline Evaluation ────────────────────────────────────────────────────
def run_pipeline_test():
    header("TEST 3 · Full Pipeline  (Code-switch + Entity Preservation)")
    from ml.evaluation.evaluate_pipeline import run_full_pipeline_eval

    manifest = "data/manifests/prism_synthetic_domain.jsonl"
    t0 = time.time()
    try:
        report = run_full_pipeline_eval(manifest_path=manifest,
                                        output_dir="ml/reports/pipeline")
        elapsed = time.time() - t0

        print(f"\n  {'Metric':<40} {'Value':>10}")
        print(f"  {'-'*52}")
        print(f"  {'Total Samples':<40} {report['total_samples']:>10}")
        cs  = report["code_switch_detection_accuracy"]
        bleu= report["translation_metrics"]["bleu"]
        chrf= report["translation_metrics"]["chrf"]
        ep  = report["semantic_entity_preservation_score"]

        cs_col = GREEN if cs >= 0.8 else (YELLOW if cs >= 0.6 else RED)
        ep_col = GREEN if ep >= 0.8 else (YELLOW if ep >= 0.6 else RED)

        print(f"  {'Code-Switch Detection Accuracy':<40} {cs_col}{cs*100:>9.1f}%{RESET}")
        print(f"  {'Translation BLEU':<40} {bleu:>10.2f}")
        print(f"  {'Translation chrF':<40} {chrf:>10.2f}")
        print(f"  {'Entity Preservation Score':<40} {ep_col}{ep*100:>9.1f}%{RESET}")

        if cs >= 0.8 and ep >= 0.7:
            ok(f"PIPELINE PASSED  ({elapsed:.1f}s)")
        else:
            warn(f"PIPELINE NEEDS IMPROVEMENT  ({elapsed:.1f}s)")
        return report

    except Exception as e:
        err(f"Pipeline test failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PRISM Model Accuracy Tests")
    parser.add_argument("--test", choices=["asr", "translation", "pipeline", "all"],
                        default="all", help="Which test suite to run (default: all)")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  PRISM  ·  Model Accuracy Test Suite{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    results = {}
    t_start = time.time()

    if args.test in ("asr", "all"):
        results["asr"] = run_asr_test()

    if args.test in ("translation", "all"):
        results["translation"] = run_translation_test()

    if args.test in ("pipeline", "all"):
        results["pipeline"] = run_pipeline_test()

    total = time.time() - t_start
    header(f"ALL TESTS COMPLETE  ({total:.1f}s total)")

    # Save combined summary
    summary_path = Path("ml/reports/accuracy_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok(f"Combined report saved → {summary_path}")
    print()


if __name__ == "__main__":
    main()
