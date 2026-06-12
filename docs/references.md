# References

Every physics claim in the tutorial notebooks (`examples/01`–`05`) links here.

## Cited works

### <a id="borgi-2024"></a>Borgi 2024 — the forward model

Role: the dislocation-contrast forward model this package implements (two-stage GO + resolution kernel).

```bibtex
@article{borgi2024,
  author  = {Borgi, S. and R{\ae}der, T. M. and Carlsen, M. A. and Detlefs, C. and Winther, G. and Poulsen, H. F.},
  title   = {Simulations of dislocation contrast in dark-field {X}-ray microscopy},
  journal = {Journal of Applied Crystallography},
  volume  = {57},
  pages   = {358--368},
  year    = {2024},
  doi     = {10.1107/S1600576724001183},
}
```

---

### <a id="borgi-2025"></a>Borgi 2025 — identification

Role: the identification approach notebook 05 demonstrates; the g·b invisibility physics appears in notebook 04.

```bibtex
@article{borgi2025,
  author  = {Borgi, S. and Winther, G. and Poulsen, H. F.},
  title   = {Individual dislocation identification in dark-field {X}-ray microscopy},
  journal = {Journal of Applied Crystallography},
  volume  = {58},
  pages   = {813--821},
  year    = {2025},
  doi     = {10.1107/S1600576725002614},
}
```

---

### <a id="poulsen-2021"></a>Poulsen 2021 — original GO model

Role: the original MATLAB geometrical-optics model this package ports and extends.

```bibtex
@article{poulsen2021,
  author  = {Poulsen, H. F. and Dresselhaus-Marais, L. E. and Carlsen, M. A. and Detlefs, C. and Winther, G.},
  title   = {Geometrical-optics formalism to model contrast in dark-field {X}-ray microscopy},
  journal = {Journal of Applied Crystallography},
  volume  = {54},
  pages   = {1555--1571},
  year    = {2021},
  doi     = {10.1107/S1600576721007287},
}
```

---

### <a id="poulsen-2017"></a>Poulsen 2017 — DFXM optics and resolution function

Role: the DFXM optics and resolution-function theory behind the reciprocal-space kernel.

```bibtex
@article{poulsen2017,
  author  = {Poulsen, H. F. and Jakobsen, A. C. and Simons, H. and Ahl, S. R. and Cook, P. K. and Detlefs, C.},
  title   = {X-ray diffraction microscopy based on refractive optics},
  journal = {Journal of Applied Crystallography},
  volume  = {50},
  pages   = {1441--1456},
  year    = {2017},
  doi     = {10.1107/S1600576717011037},
}
```

---

### <a id="hirth-lothe-1982"></a>Hirth & Lothe 1982 — dislocation theory

Role: the dislocation displacement fields used by the direct-space model.

```bibtex
@book{hirthlothe1982,
  author    = {Hirth, J. P. and Lothe, J.},
  title     = {Theory of Dislocations},
  edition   = {2},
  publisher = {Wiley},
  address   = {New York},
  year      = {1982},
}
```

---

### <a id="carlsen-2022"></a>Carlsen 2022 — wave-optics DFXM model

Role: wave-optics DFXM model; provenance of the condenser-aperture clip (±140 µrad zeta\_v) both resolution backends apply.

```bibtex
@article{carlsen2022,
  author  = {Carlsen, M. and Detlefs, C. and Yildirim, C. and R{\ae}der, T. and Simons, H.},
  title   = {Simulating dark-field {X}-ray microscopy images with wave front propagation techniques},
  journal = {Acta Crystallographica Section A},
  volume  = {78},
  pages   = {482--490},
  year    = {2022},
  doi     = {10.1107/S2053273322007379},
  eprint  = {2201.07549},
  archivePrefix = {arXiv},
}
```

---

### <a id="darkmod"></a>darkmod — independent GO-family comparison code

Role: independent GO-family DFXM implementation used for cross-comparison.

```bibtex
@misc{darkmod2025,
  author        = {Henningsson, A. and Borgi, S. and Winther, G. and El-Azab, A. and Poulsen, H. F.},
  title         = {Towards Interfacing Dark-Field {X}-ray Microscopy to Dislocation Dynamics Modeling},
  year          = {2025},
  eprint        = {2503.22022},
  archivePrefix = {arXiv},
  doi           = {10.1016/j.jmps.2025.106277},
}
```

---

## Where each is used

| Reference | Notebooks | Docs |
|---|---|---|
| [Borgi 2024](#borgi-2024) | 01, 02, 03, 04 | README, physics.md |
| [Borgi 2025](#borgi-2025) | 04 (g·b criterion), 05 | README |
| [Poulsen 2021](#poulsen-2021) | 02 | README |
| [Poulsen 2017](#poulsen-2017) | 02 | physics.md |
| [Hirth & Lothe 1982](#hirth-lothe-1982) | 03 | physics.md |
| [Carlsen 2022](#carlsen-2022) | 02 | model-taxonomy.md, resolution-backend-comparison.md |
| [darkmod](#darkmod) | — | model-taxonomy.md, recovery-plan-darkmod-vs-go.md |
