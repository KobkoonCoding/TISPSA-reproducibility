# Data sources, provenance and licenses

All files in this directory are redistributed **unchanged** from public repositories
(raw files) together with the preprocessed versions produced by `tispsa/data.py`
(CT and MRI: grayscale, center square crop; fundus: green channel, 700x700 window around the optic disc;
all resized to 256x256 with Lanczos filtering).
SHA-256 hashes of every file are listed in `SHA256SUMS`; `python -m tispsa.data`
re-downloads the raw files from the URLs below and verifies them.

## Which file is which: original name, package file and hash

Every hash line of `SHA256SUMS` is preceded by a comment giving the same information, so a
hash can be traced to its origin without leaving that file.

| Original file (in the source repository) | Package file | SHA-256 |
|---|---|---|
| `images/jkms-35-e79-g001-l-d.jpg` — `ieee8023/covid-chestxray-dataset` | `images/raw/ct_raw.jpg` -> `images/ct.png` | `645f50bcaa1ec9bdf4b74aae8c9e9e91cb3f39d33870364dbd7cea1c248db7ae` |
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
| Chest CT (axial) | `images/raw/ct_raw.jpg` | COVID-19 Image Data Collection [1], file `images/jkms-35-e79-g001-l-d.jpg` (Figure 1D of J. Korean Med. Sci. 35(6):e79, 2020, doi:10.3346/jkms.2020.35.e79) | CC BY-NC 4.0 (article, Crossref record of doi:10.3346/jkms.2020.35.e79); CC BY-NC-SA (as recorded in `metadata.csv`) | Lim et al. (2020), J. Korean Med. Sci. 35(6):e79 |
| Brain MRI (axial) | `images/raw/mri_raw.jpg` | Brain Tumor Classification (MRI) dataset [2], file `Testing/no_tumor/image(2).jpg` (GitHub mirror of the Kaggle dataset) | MIT (Kaggle dataset page and API, version 3, checked 3 October 2026) | Bhuvaji et al. (2020) |

[1] J. P. Cohen, P. Morrison, L. Dao, COVID-19 image data collection, arXiv:2003.11597 (2020).
    https://github.com/ieee8023/covid-chestxray-dataset
[2] S. Bhuvaji, A. Kadam, P. Bhumkar, S. Dedge, S. Kanchan, Brain Tumor Classification (MRI), Kaggle (2020).
    doi:10.34740/KAGGLE/DSV/1183165 ; https://github.com/sartajbhuvaji/brain-tumor-classification-dataset

The CT image is licensed for non-commercial use (CC BY-NC 4.0 for the article, CC BY-NC-SA in the dataset
metadata): it is redistributed here for non-commercial research use, with attribution, under the same license.
It is used for the numbers of Tables 2, 3 and 6 and the CT panels of Figures 1, 2 and 3. The arrays `CT_*` in
`expected_results/deblur_figimages.npz` are derived from it and carry the same license.

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
