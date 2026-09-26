# Statistical Anomaly Detection for a Multi-Area Power Network

Final Year Project (NTU): detecting faults and stealthy cyber-attacks in a
networked power system with a distributed state estimator and a statistical
test on its residuals.

## How the pieces fit

```
 Phase B: plant + estimator          Phase C (planned)                 Phase A: detector
 ─────────────────────────           ─────────────────────             ─────────────────
 power network model   ──gains──▶    simulate network, run     ──z──▶  is z still i.i.d. N(0,1)?
 per-area estimator                  estimators, normalise              → v, psi, alarm
                                     residuals z = Γ^{-1/2}(y − ŷ)
```

| Phase | Script | Objective | Output |
|---|---|---|---|
| **A** | [`phase_a.py`](phase_a.py) | Build the statistical detector and check it on synthetic residuals where the correct answer is known. | `detector_stats.npz` |
| **B** | [`phase_b.py`](phase_b.py) | Build the 4-area power network model (plus a post-fault version) and design one estimator per area. | `estimator_gains.npz` |
| C | not yet written | Connect B to A: simulate the network, feed the normalised residuals into the detector. | — |

If the estimator from Phase B is working and nothing is wrong, its normalised
residual is white Gaussian noise. Phase A's detector checks exactly that, so a
fault (Phase B's inertia drop in area 5) or an attack shows up as an alarm.

## Setup and running

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python phase_b.py               # fast
python phase_a.py               # slower: k-means + several 100 000-step detector runs
```

Each script prints a numbered list of validation checks, each marked `[OK]` or
`[FAIL]`, then saves its `.npz` file next to the script. The `.npz` files are
generated, so they are not committed (see `.gitignore`).

## Documentation

- [docs/phase_a.md](docs/phase_a.md): detector objective, method, symbols, how to read the output
- [docs/phase_b.md](docs/phase_b.md): plant model, estimator design, symbols, how to read the output

## References

Numbering matches the `[n]` citations in the code.

1. Yan, Fu & Seron, *Anomaly Detection for Stochastic Networked Cyber-Physical Systems: A Statistical Approach*, IEEE ICCA 2024.
2. Boem et al., *Distributed Fault Detection for Interconnected Large-Scale Systems: A Scalable Plug & Play Approach*, IEEE TCNS 6(2), 2019.
3. Marelli, Sui, Fu & Lu, *Statistical Approach to Detection of Attacks for Stochastic Cyber-Physical Systems*, IEEE TAC 66(2), 2021.
5. Riverso & Ferrari-Trecate, *HyCon2 Benchmark: Power Network System*, arXiv:1207.2000.
