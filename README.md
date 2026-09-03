# vTR-qMRI
Multi-compartment relaxometry myelin water imaging (MCR-MWI) for **variable flip angle / variable TR (VFA-VTR)** multi-echo gradient-echo data.
This extends [`gpuMCRMWI`](https://github.com/kschan0214/gacelle) from the GACELLE toolbox, where all flip angles share one TR and one set of echo times. Here every acquisition *k* has its own flip angle `fa(k)`, repetition time `tr(k)` and echo-time vector `te{k}`.

| File | |
|---|---|
| `gpuMCRMWI_VFAVTR.m` | main fitting class |
| `despot1_VTR.m` | variable-TR DESPOT1, used for the S0 / R1 starting points and the fitting mask |

## Requirements

- MATLAB (tested with R2024b)
- Parallel Computing, Deep Learning and Statistics and Machine Learning Toolboxes
- A CUDA-capable GPU
- [GACELLE](https://github.com/kschan0214/gacelle) — tested with commit `5e251b8` (2026-08-06)

```matlab
run('/path/to/gacelle/addpath_gacelle.m');
addpath('/path/to/vTR-qMRI');
```

## Usage

```matlab
% protocol: one entry per acquisition (TE, TR in s; FA in degrees)
te = { te_acq1, te_acq2, te_acq3 };     % cell, one echo-time vector per acquisition
tr = [ 11e-3 ; 28e-3 ; 44e-3 ];
fa = [ 5 ; 20 ; 70 ];

obj = gpuMCRMWI_VFAVTR(te, tr, fa, fixed_params);
out = obj.estimate(data, mask, extraData, fitting);
```

**`data` is 4D**, `[x, y, z, Nmeas]`, with acquisitions concatenated along the 4th dimension and echoes within each acquisition. This differs from the 5D `[x,y,z,TE,FA]` layout of `gpuMCRMWI`, since acquisitions may have different numbers of echoes.

| Input | |
|---|---|
| `mask` | 3D logical fitting mask |
| `extraData.b1` | 3D B1+ map, ratio of actual to nominal flip angle |
| `extraData.freqBKG` | total field in **ppm**, one map per acquisition |
| `extraData.pini` | 3D initial phase [rad] (optional) |
| `fixed_params` | `B0`, `B0dir`, `rho_mw`, `E`, `x_i`, `x_a`, `t1_mw`, `thres_R2star`, `thres_T1` (all optional) |
| `fitting` | askAdam options plus `isComplex`, `isFitExchange`, `isEPG`, `DIMWI.*` (see `help gpuMCRMWI_VFAVTR/fit`) |

`out.final.<param>` and `out.min.<param>` hold the maps at the last iteration and at the minimum loss; `out.mask` is the mask actually fitted.

## Citation

Please cite the underlying MCR-MWI method:

- Chan K-S, Marques JP. Multi-compartment relaxometry and diffusion informed myelin water imaging — promises and challenges of new gradient echo myelin water imaging methods. *NeuroImage* 2020;221:117159.
- Chan K-S, Chamberland M, Marques JP. On the performance of multi-compartment relaxometry for myelin water imaging (MCR-MWI) — test-retest repeatability and inter-protocol reproducibility. *NeuroImage* 2023;266:119824.

## Licence

GNU General Public License v3, see [LICENSE](LICENSE). `gpuMCRMWI_VFAVTR.m` and `despot1_VTR.m` are derived from GPL-3 code in GACELLE, Copyright (c) Kwok-Shing Chan, Massachusetts General Hospital.

## Contact

Samuel Melke Gebremedhin — samuel.gebremedhin@donders.ru.nl
Donders Centre for Cognitive Neuroimaging, Radboud University, Nijmegen
