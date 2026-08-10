"""
הצינור כולו, מהורדת הנתונים ועד האיורים והבדיקות.

  python run_all.py                # הכל, מההתחלה
  python run_all.py --list         # רשימת השלבים ומה כל אחד מייצר
  python run_all.py --from model   # להמשיך משלב מסוים
  python run_all.py --only figures

הקוד יושב ב-code/, הנתונים הגולמיים ב-data/, וכל מה שנוצר נכתב ל-work/.
ההפרדה הזו היא כל מה שצריך כדי להריץ במחשב חדש: אחרי download_data.py אין
בתיקייה שום קובץ שנוצר במחשב אחר.

**סדר השלבים אינו שרירותי.** keyness חייב לרוץ לפני בניית הקורפוס, משום
ש-bounding.py קורא את keyness_word_weights.csv - ובהיעדרו אחד מכללי החיתוך
פשוט אינו פועל, בלי הודעת שגיאה, והקורפוס יוצא אחר. זו התקלה הכי שקטה
בצינור הזה, ולכן היא כתובה כאן ולא רק ב-README.
"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE, DATA, WORK = (os.path.join(ROOT, d) for d in ("code", "data", "work"))

# הקבצים ש-themes.py כותב לתיקייה הנוכחית, ושהאיורים והבדיקות קוראים
# מ-final_refit/. עד עכשיו ההעתקה נעשתה ביד, ולכן היא לא הייתה משוחזרת
REFIT_FILES = ["topic_shares_by_decade.csv", "topic_labels.csv",
               "topic_lift_by_decade.csv", "topic_prevalence_by_decade.csv",
               "decade_digest.csv", "artifact_share_by_decade.csv"]


def sh(*args, **kw):
    env = dict(os.environ, PYTHONPATH=CODE)
    r = subprocess.run(args, cwd=WORK, env=env, **kw)
    if r.returncode:
        raise SystemExit(f"stage failed: {' '.join(args)}")


def py(script, *args):
    sh(sys.executable, os.path.join(CODE, script), *args)


def snap_refit():
    """העתקת פלטי ההתאמה אל final_refit/ - השלב שהיה חסר מהצינור."""
    dest = os.path.join(WORK, "final_refit")
    os.makedirs(dest, exist_ok=True)
    for f in REFIT_FILES:
        src = os.path.join(WORK, f)
        if not os.path.exists(src):
            raise SystemExit(f"themes.py did not produce {f}")
        shutil.copy2(src, os.path.join(dest, f))
    print(f"  copied {len(REFIT_FILES)} files into work/final_refit/")


def stage_download():
    py("download_data.py")


def stage_keyness():
    # ההתאמה בין גודריידס ל-CMU, ואז טבלת משקלי הרגיסטר שה-bounding צורך
    py("keyness.py")
    sh(sys.executable, "-c",
       "import keyness; keyness.export_word_weights()")


def stage_corpus():
    py("build_bounded_corpus.py")


def stage_model():
    py("themes.py")
    snap_refit()


def stage_figures():
    py("writeup_figures.py")
    py("trends_report.py")


def stage_checks():
    # verify_writeup רץ *אחרי* stability_curves, לא לפניו: ארבע מתוך 36
    # הבדיקות קוראות את all_topic_stability.csv, ובריצה נקייה הקובץ עדיין
    # לא קיים - אז הן דולגו בשקט והדוח הראה 32/32 במקום 36/36
    py("stability_curves.py")
    py("stability.py")
    py("sf_stability.py")
    py("evaluate_bounding.py")
    py("roc_filtering.py")
    py("raw_vs_renorm.py")
    py("verify_writeup.py")


STAGES = [
    ("download", stage_download, "data/*.json.gz, data/booksummaries.txt"),
    ("keyness", stage_keyness, "keyness_matched.pkl, keyness_word_weights.csv"),
    ("corpus", stage_corpus, "themes_corpus_bounded.pkl"),
    ("model", stage_model, "final_refit/*.csv"),
    ("figures", stage_figures, "wfig1-wfig3, topic_trends_v2.pdf"),
    ("checks", stage_checks, "the validation numbers quoted in the writeup"),
]


def main():
    names = [n for n, _, _ in STAGES]
    if "--list" in sys.argv:
        for n, _, produces in STAGES:
            print(f"  {n:9} -> {produces}")
        return
    todo = names
    if "--from" in sys.argv:
        todo = names[names.index(sys.argv[sys.argv.index("--from") + 1]):]
    if "--only" in sys.argv:
        todo = [sys.argv[sys.argv.index("--only") + 1]]

    os.makedirs(WORK, exist_ok=True)
    for name, fn, _ in STAGES:
        if name not in todo:
            continue
        print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}")
        t = time.time()
        fn()
        print(f"-- {name} finished in {time.time() - t:.0f}s")
    print("\nall requested stages completed")


if __name__ == "__main__":
    main()
