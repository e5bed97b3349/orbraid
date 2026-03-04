# orbraid

**3D choreographic N-body orbits: point-group classification, knot invariants, and stability analysis.**

A choreographic orbit is a periodic solution to the Newtonian N-body problem in which all bodies follow a single closed curve, equally spaced in phase. While the two-dimensional case has been surveyed extensively (Simó 2002), fully three-dimensional choreographies remain largely unexplored.

This project organises the search by the point-group symmetry of the orbit — importing the Schönflies classification from molecular symmetry into gravitational dynamics — and characterises each discovered orbit by the knot type of its spacetime trace. The result is a correspondence between symmetry class, knot invariant, and action value that draws explicit connections to the literature on mechanically synthesised molecular knots.

## Mathematical setting

All N bodies have equal mass. We seek critical points of the Lagrangian action

$$\mathcal{A}[\gamma] = \frac{1}{2}\int_0^1 |\dot\gamma|^2\,dt + \frac{1}{N}\sum_{k=1}^{N-1}\int_0^1 \frac{dt}{|\gamma(t) - \gamma(t+k/N)|}$$

over the choreographic subspace, constrained to the fixed-point subspace Fix(G) of a target point group G ⊂ O(3) × ℤ/Nℤ. By the Palais symmetric criticality principle, minimisers of the restricted functional are genuine N-body solutions.

The spacetime trace of each orbit is a closed curve in S³ whose knot type — characterised by the Alexander and HOMFLY-PT polynomials — is a topological invariant of the choreography.

## Repository structure

```
orbraid/
├── docs/
│   ├── rdd/            Research Design Document (v0.1)
│   ├── compspec/       Computational Specification (v0.1)
│   └── mathframe/      Mathematical Framework (v0.1)
├── src/
│   └── orbraid/        Python source (JAX implementation)
├── catalogue/          HDF5 orbit catalogue (generated)
└── figures/            Renders, animations, publication figures
```

## Documents

| Document | Description | Version |
|----------|-------------|---------|
| [RDD](docs/rdd/rdd.pdf) | Research design, motivation, methodology overview | v0.1 |
| [MathFrame](docs/mathframe/mathframe.pdf) | Full mathematical framework: symmetry constraints, existence theory, knot theory, Floquet–Krein stability | v0.1 |
| [CompSpec](docs/compspec/compspec.pdf) | Computational specification: JAX implementation, algorithms, catalogue schema | v0.1 |

## Status

Active development. See [Issues](../../issues) and [Milestones](../../milestones) for the full project backlog.

| Milestone | Status |
|-----------|--------|
| M0: Environment & validation | 🔲 Not started |
| M1: C₃ᵥ / D₃ search (N=3) | 🔲 Not started |
| M2: All N=3 and N=4 groups | 🔲 Not started |
| M3: Knot pipeline | 🔲 Not started |
| M4: Stability analysis | 🔲 Not started |
| M5: Tetrahedral and octahedral | 🔲 Not started |
| M6: Icosahedral attempt | 🔲 Not started |
| M7: Visualisation pipeline | 🔲 Not started |
| M8: Catalogue, toolkit, and paper | 🔲 Not started |

## Requirements

```
jax[cuda12]>=0.4.25
optax>=0.2.2
scipy>=1.13
snappy>=3.1
pyvista>=0.44
ffmpeg-python>=0.2
numpy>=1.26
h5py>=3.11
```

## Citation

*Preprint forthcoming.* Until then, please cite the repository directly:

```bibtex
@software{orbraid2026,
  author  = {e5bed97b3349},
  title   = {orbraid: 3D choreographic N-body orbits},
  year    = {2026},
  url     = {https://github.com/e5bed97b3349/orbraid}
}
```

## Licence

MIT — see [LICENSE](LICENSE).
