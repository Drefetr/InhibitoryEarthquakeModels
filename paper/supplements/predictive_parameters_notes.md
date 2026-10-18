# Saved predictive fits: table proposal

All values come from accepted iteration-27 selected fits. Display values are rounded; full precision, raw solver lineage and source pointers are in `fit_pod_parameters.json`. No fitting or likelihood evaluation was run.

| Origin | Family | a | d | n | eta | r | Kernel |
|---|---|---:|---:|---:|---:|---:|---|
| 50% | Omori | -4.712517 | -0.457071 | 0.685640 | 0.000000 | 0.4 | c=0.012995, p=1.113368 |
| 50% | Mittag--Leffler | -4.458572 | -0.252150 | 0.455824 | 0.000000 | 0.4 | alpha=0.5, tau=1 |
| 50% | Shared inhibition | -2.196743 | 1.789757 | 0.481700 | 1.646714 | 0.4 | alpha=0.5, tau=1 |
| 65% | Omori | -4.291554 | 0.048036 | 0.570076 | 0.000000 | 0.7 | c=0.020798, p=1.198604 |
| 65% | Mittag--Leffler | -4.188810 | 0.147441 | 0.497996 | 0.000000 | 0.7 | alpha=0.5, tau=1 |
| 65% | Shared inhibition | -3.841844 | 0.492682 | 0.503490 | 0.211710 | 0.7 | alpha=0.5, tau=1 |
| 80% | Omori | -4.394252 | -0.196162 | 0.561216 | 0.000000 | 0.7 | c=0.006972, p=1.155845 |
| 80% | Mittag--Leffler | -4.282575 | -0.056651 | 0.470292 | 0.000000 | 0.7 | alpha=0.5, tau=1 |
| 80% | Shared inhibition | -4.103408 | 0.127638 | 0.473071 | 0.105254 | 0.7 | alpha=0.5, tau=1 |


The origins use strict training prefixes ending at 12197, 15856.1 and 19515.2 days, with 301, 504 and 659 training events; test counts are 203, 155 and 254. The training-only exponential mark rates are respectively 3.1016219112585786, 3.1494882081661726 and 3.021907322176272 per magnitude unit. They are common to all five families and frozen during each test window.

Here z(t)=2t/24394-1 and the background is exp(a+d*z(t)-eta*g(t)) events/day. Kernel scales c and tau use days. The coefficient n multiplies normalized mark productivity (1-r)*exp(beta_train*r*excess). The shared stress uses fixed gamma=0.75*log(10) and weights exp(gamma*(excess-1)); eta=0 is fixed for the two triggering comparators. Omori c and p were selected by continuous bounded optimization with a retained coarse-grid floor; they are not grid winners. The Mittag-Leffler alpha and tau and the productivity ratio r were selected on the frozen grids. All six selected Mittag-Leffler kernels have alpha=0.5 at the lower searched alpha endpoint; tau=1 is an interior grid value. Ratio r=0.4 at the first origin is the lower searched ratio endpoint; r=0.7 at the later origins is interior. These are derived endpoint checks, not archived flags or uncertainty statements. All selected estimated continuous coordinates are strictly inside their recorded bounds under exact comparison. Historical boundary flags, near-boundary tolerances and standard errors were not saved.

The machine-readable table also retains the constant and loglinear Poisson comparators. In the constant family d=0 is fixed; n=eta=0 are fixed in both Poisson families. The KernelSpec fields p=1.3 for Mittag-Leffler and alpha=0.5 for Omori are inactive serialization defaults and are not fitted kernel coordinates. The saved training objective excludes the common mark-density term.

The larger fitted eta at the first origin describes an exploratory recipe selected on previously seen data. It does not identify a geological inhibition mechanism. The new table leaves the reused-period and unequal-kernel-flexibility qualifications intact. The saved shared-minus-Mittag-Leffler score total remains -17.622859364794834; Omori-minus-Mittag-Leffler remains +12.458120885646906. The near-zero second-fold contrast is not parameter equivalence or a calibrated adequacy finding. No new confidence intervals or event-count threshold are inferred.
