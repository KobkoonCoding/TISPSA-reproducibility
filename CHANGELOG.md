# Changelog

## 3.6.0

* The CT test image of Section 4.2 is now an axial chest CT slice by Mikael Häggström (Wikimedia Commons, CC0),
  so that every image of the package and of the paper can be reused without permission. All CT results (Tables 2,
  3 and 6, Figures 1-3, the selected λ_TV) were recomputed.
* Table 1 lists TISPSA first in each problem; Table 4 gives each dataset name once.

## 3.5.1

Table 3 (`tables/table3_tolerance.tex`) lists the iterations and the evaluations of T for all three methods. The
numbers are unchanged.

## 3.5.0

Figure 1 shows all three test images (fundus, CT and MRI) and the images restored by all three methods (TISPSA and
the method of Boţ et al. with λ_n = 1 and λ_n = 1.4). `expected_results/deblur_figimages.npz` stores these images.
All tables and all other numbers are unchanged.

## 3.4.0 (first public release)

Package that reproduces the tables, figures and numbers of Section 4 of the paper "A double inertial S-iteration
algorithm with Tikhonov regularization for common fixed points and forward-backward splitting, with applications to
medical image deblurring and medical data classification".
