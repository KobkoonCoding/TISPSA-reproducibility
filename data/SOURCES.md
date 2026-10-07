# Data sources, provenance and licenses

All files in this directory are redistributed **unchanged** from public repositories
(raw files) together with the preprocessed versions produced by `tispsa/data.py`
(CT: grayscale, 780x780 window around the thorax; MRI: grayscale, center square crop; fundus: green channel,
700x700 window around the optic disc;
all resized to 256x256 with Lanczos filtering).
SHA-256 hashes of every file are listed in `SHA256SUMS`; `python -m tispsa.data`
re-downloads the raw files from the URLs below and verifies them.

## Which file is which: original name, package file and hash

Every hash line of `SHA256SUMS` is preceded by a comment giving the same information, so a
hash can be traced to its origin without leaving that file.

| Original file (in the source repository) | Package file | SHA-256 |
|---|---|---|
| `File:High-resolution computed tomograph of a normal thorax, axial plane (38).jpg` — Wikimedia Commons | `images/raw/ct_raw.jpg` -> `images/ct.png` | `7d84f4c5491e0efac6f9db5a9e7af4ab621264976d6855f269584c807af4c23a` |
| `Testing/no_tumor/image(2).jpg` — `sartajbhuvaji/brain-tumor-classification-dataset` | `images/raw/mri_raw.jpg` -> `images/mri.png` | `9cb0bd6ba354991ae933b55af58ca040f3b8db01add876aba09159d7badd3599` |
| `skimage.data.retina()` (scikit-image, no file downloaded) | `images/fundus.png` | `e9979ece1c00df2e753a34719297a77198d718c2488c9cf2b013f4206f5fa02d` |
| `pima-indians-diabetes.data.csv` — `jbrownlee/Datasets` | `tabular/pima-indians-diabetes.csv` | `6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af` |
| `processed.cleveland.data` — UCI Heart Disease | `tabular/processed.cleveland.data` | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |

The hashes of `images/ct.png` and `images/mri.png`, which are produced from
the raw files by `tispsa/data.py`, are listed in `SHA256SUMS` together with the raw file each of
them is derived from.

To check the bundled copies without downloading or overwriting anything:

```bash
sha256sum -c SHA256SUMS            # from this directory
python -m tispsa.data --verify     # from the package root, same check
```

`python -m tispsa.data` (without `--verify`) re-downloads the raw files from the URLs in
`SHA256SUMS`, **overwrites** the bundled copies and then verifies them.

## Medical images (Section 4.2)

| Paper name | Raw file | Source | License | Attribution |
|---|---|---|---|---|
| Retinal fundus | none (generated from `skimage.data.retina()`, bundled with scikit-image); preprocessed file `images/fundus.png` = green channel, 700x700 window around the optic disc (box (0,307,700,1007)), resized to 256x256 | Häggström, M. (2014), "Medical gallery of Mikael Häggström 2014", WikiJournal of Medicine 1(2):8, doi:10.15347/wjm/2014.008 (file `Fundus_photograph_of_normal_left_eye.jpg` on Wikimedia Commons) | CC0 1.0 (public domain dedication) | Mikael Häggström |
| Chest CT (axial) | `images/raw/ct_raw.jpg`; preprocessed file `images/ct.png` = grayscale, 780x780 window (370,140,1150,920) that leaves out the text and the scout image of the screenshot, resized to 256x256 | Häggström, M. (2017), "High-resolution computed tomograph of a normal thorax, axial plane (38)", Wikimedia Commons, https://commons.wikimedia.org/wiki/File:High-resolution_computed_tomograph_of_a_normal_thorax,_axial_plane_(38).jpg (own work; slice 38 of a high-resolution chest CT of a 37-year-old man, 2017-05-20; SHA-1 9faed60d48cd302d37a60aff0051733dd59949a8 as recorded by Commons) | CC0 1.0 (public domain dedication) | Mikael Häggström |
| Brain MRI (axial) | `images/raw/mri_raw.jpg` | Brain Tumor Classification (MRI) dataset [1], file `Testing/no_tumor/image(2).jpg` (GitHub mirror of the Kaggle dataset) | MIT (Kaggle dataset page and API, version 3, checked 3 October 2026) | Bhuvaji et al. (2020) |

[1] S. Bhuvaji, A. Kadam, P. Bhumkar, S. Dedge, S. Kanchan, Brain Tumor Classification (MRI), Kaggle (2020).
    doi:10.34740/KAGGLE/DSV/1183165 ; https://github.com/sartajbhuvaji/brain-tumor-classification-dataset

All three images can be reused without permission: the fundus and CT images are dedicated to the public domain
(CC0) and the MRI image is under the MIT license. The CT image replaces, from version 3.6.0 on, the CT image of the
COVID-19 Image Data Collection used in earlier versions, which was licensed for non-commercial use only.

## Clinical datasets (Section 4.3)

| Paper name | File | Source | License |
|---|---|---|---|
| WDBC | loaded via `sklearn.datasets.load_breast_cancer` (bundled with scikit-learn) | UCI Machine Learning Repository, doi:10.24432/C5DW2B | CC BY 4.0 |
| Pima | `tabular/pima-indians-diabetes.csv` | Smith et al. (1988); copy from https://github.com/jbrownlee/Datasets (`pima-indians-diabetes.data.csv`) | public domain (NIDDK / former UCI dataset) |
| Heart | `tabular/processed.cleveland.data` | UCI Heart Disease (Cleveland), doi:10.24432/C52P4X; downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data (303 rows, no header, 13 attributes + `num`) | CC BY 4.0 |

### Heart: which file is used, and how the positive class is defined

The Heart results in Section 4.3 are computed from `tabular/processed.cleveland.data`,
the file distributed by the UCI Machine Learning Repository under the DOI cited in the
paper (10.24432/C52P4X).  It is loaded by `tispsa.data.load_heart()`, which

* reads the 14 unnamed columns as `age, sex, cp, trestbps, chol, fbs, restecg, thalach,
  exang, oldpeak, slope, ca, thal, num`;
* drops the 6 records with a missing value (`?`) in `ca` or `thal`, leaving **297** of the
  303 records;
* defines the **positive class as the presence of disease, `num > 0`** (**137** cases,
  46.13%).
