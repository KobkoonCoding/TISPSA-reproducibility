"""Data: benchmark medical images (Section 4.2) and clinical datasets (Section 4.3).

All raw files are bundled in ``data/`` (see ``data/SOURCES.md`` for provenance and
licenses).  ``download_all()`` re-fetches them from the original public repositories
and checks the SHA-256 hashes, so that the bundled copies can be verified.
"""
import hashlib, os, urllib.request
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

IMAGES = {   # name: (raw file, source URL)
    "CT":    ("ct_raw.jpg",   "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/jkms-35-e79-g001-l-d.jpg"),
    "MRI":   ("mri_raw.jpg",  "https://raw.githubusercontent.com/sartajbhuvaji/brain-tumor-classification-dataset/master/Testing/no_tumor/image(2).jpg"),
}
TABULAR = {
    "pima-indians-diabetes.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "processed.cleveland.data":  "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
}

# UCI Heart Disease (Cleveland), doi:10.24432/C52P4X.  The file has no header; the 13
# attributes below are the ones distributed in ``processed.cleveland.data`` and ``num`` is
# the diagnosis (0 = no disease, 1-4 = disease).  Missing values are encoded as "?" and
# occur only in ``ca`` and ``thal``.
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
              "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_heart():
    """UCI Cleveland heart-disease data: 297 complete cases, positive class = disease (num > 0).

    Rows with a missing (``?``) value in ``ca`` or ``thal`` are dropped, which leaves
    297 of the 303 records, 137 of them positive (46.13%).
    """
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "tabular", "processed.cleveland.data"), header=None, names=HEART_COLS)
    keep = ~(df[["ca", "thal"]].astype(str) == "?").any(axis=1)
    df = df[keep]
    X = df[HEART_COLS[:-1]].to_numpy(dtype=float)
    y = (df["num"].to_numpy(dtype=float) > 0).astype(int)
    return X, y


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def preprocess_image(raw_path, out_path, size=256):
    """Grayscale, center square crop, resize to size x size (Lanczos), save PNG."""
    im = Image.open(raw_path).convert("L")
    w, h = im.size; s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.LANCZOS)
    im.save(out_path)


FUNDUS_CROP = (0, 307, 700, 1007)   # 700x700 window around the optic disc (left, upper, right, lower)


def make_fundus(out_path, size=256):
    """Retinal fundus image: green channel of skimage.data.retina() (CC0, Haggstrom 2014),
    700x700 crop around the optic disc, resized to size x size (Lanczos)."""
    import skimage.data
    g = Image.fromarray(skimage.data.retina()[:, :, 1])
    g.crop(FUNDUS_CROP).resize((size, size), Image.LANCZOS).save(out_path)


def load_image(name):
    """Return the preprocessed 256x256 image with intensities in [0,1]."""
    return np.asarray(Image.open(os.path.join(DATA, "images", f"{name.lower().replace('-', '')}.png")).convert("L"), dtype=float) / 255.0


def read_sums():
    """Parse data/SHA256SUMS -> {relative path: sha256}.  Comment ('#') and blank lines are skipped;
    the comments record the original file name and URL of every entry."""
    sums = {}
    with open(os.path.join(DATA, "SHA256SUMS")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            h, name = line.split(None, 1)
            sums[name.strip()] = h
    return sums


def verify_local():
    """Check every file listed in data/SHA256SUMS against its recorded hash.  Nothing is
    downloaded and nothing is overwritten.  Returns True if all files match."""
    ok = True
    for name, want in read_sums().items():
        path = os.path.join(DATA, *name.split("/"))
        if not os.path.exists(path):
            print(f"{name:36s} MISSING"); ok = False; continue
        got = sha256(path)
        good = got == want
        ok &= good
        print(f"{name:36s} {'OK' if good else 'MISMATCH'}  {got}")
    print("All bundled data files match data/SHA256SUMS." if ok else "Some files do NOT match data/SHA256SUMS.")
    return ok


def download_all(verify=True):
    """Re-download the raw files from their public sources and verify them against data/SHA256SUMS.

    WARNING: this overwrites the bundled copies in data/.  Use verify_local()
    (``python -m tispsa.data --verify``) to check them without downloading."""
    sums = read_sums()
    for name, (fname, url) in IMAGES.items():
        dst = os.path.join(DATA, "images", "raw", fname)
        urllib.request.urlretrieve(url, dst)
        print(f"downloaded {fname}", "OK" if (not verify or sha256(dst) == sums[f"images/raw/{fname}"]) else "HASH MISMATCH")
        preprocess_image(dst, os.path.join(DATA, "images", f"{name.lower().replace('-', '')}.png"))
    make_fundus(os.path.join(DATA, "images", "fundus.png"))
    for fname, url in TABULAR.items():
        dst = os.path.join(DATA, "tabular", fname)
        urllib.request.urlretrieve(url, dst)
        print(f"downloaded {fname}", "OK" if (not verify or sha256(dst) == sums[f"tabular/{fname}"]) else "HASH MISMATCH")


if __name__ == "__main__":
    import sys
    if "--verify" in sys.argv[1:]:
        sys.exit(0 if verify_local() else 1)          # check the bundled files, download nothing
    download_all()                                    # re-download the raw files and overwrite them
