# Generative Design Tool: User Stories and Requirements

Status: draft v2, open questions resolved · Last updated: 2026-10-31

## 1. Product goal

**Context.** The tool is for aerospace parts made of steel, aluminium and titanium. The parts can have any purpose. The minimum viable product (MVP) is a 2D desktop application built in Python.


The tool takes a part's constraints (design envelope, connection points, keep-out zones, loads, supports, material budget) and produces a topology-optimized part. It also produces a **certificate**: a mathematical proof that no design on the same mesh, under the same loads and material budget, can be stiffer by more than a stated margin (the **optimality gap**).

Machine learning speeds up the workflow and proposes candidate designs. ML output is never trusted directly: every design the tool reports has been checked by the physics optimizer and the verifier.

```
constraints → ML first guess → physics optimizer → verifier → certified part
```

## 2. Definitions

| Term | Meaning |
| --- | --- |
| Design envelope | The bounding region the part may occupy, discretized into a mesh of elements. |
| Preserved region | Elements that must be solid, such as connection points and mounting faces. |
| Keep-out region | Elements that must be empty, such as space for bolts, cables or other parts. |
| Load case | A set of forces applied to the part together with its supports (fixed degrees of freedom). |
| Material budget | The maximum fraction of the design envelope's volume the part may use. |
| Compliance | `c(x) = fᵀu(x)`, where `K(x)u = f`. It measures how much the part deflects under load; lower means stiffer. |
| Binary design | Every element is fully solid (1) or empty (0). Only binary designs correspond to real parts. |
| Relaxed problem | The same problem with densities in `[0, 1]` and stiffness linear in density. It is convex, and every binary design is feasible in it. |
| Lower bound (LB) | A value that no feasible design's compliance can go below, computed from the relaxed problem. |
| Optimality gap | `(c(x_final) − LB) / LB`. It is a proven upper bound on how far the design is from the best possible design on this mesh and model. |
| Certificate | A report holding the problem, the design, the upper bound, the lower bound, the gap, the feasibility checks and the numerical tolerances, which anyone can re-check independently. |
| Limit load | The largest load the part is expected to see in service. |
| Ultimate load | Limit load × factor of safety. The factor of safety is 1.5 unless otherwise specified (14 CFR § 25.303). |
| Margin of safety (MS) | `allowable stress / (factor × applied stress) − 1`. It must be ≥ 0. |

## 3. Personas

- **Design engineer (Dana).** Designs brackets and mounts. Wants a lightweight, buildable part fast and needs proof that it is good enough to put in front of a review.
- **Verification engineer (Viktor).** Reviews designs before release. Trusts nothing he cannot re-check. Wants every claim backed by data he can recompute.
- **ML developer (Mira).** Builds and trains the models that speed up the optimizer. Needs data, benchmarks and a way to show that a model actually helps.

## 4. User stories

Priorities use MoSCoW: **M** must, **S** should, **C** could, **W** won't (this version).

### Epic A: Problem setup

**US-A1 (M):** As Dana, I want to define the design envelope and mesh resolution, so that the optimizer knows where material may go.
- Given a rectangular envelope and a resolution, the tool creates a structured mesh and shows it.

**US-A2 (M):** As Dana, I want to mark preserved regions (connection points), so that the part always connects where it has to.
- Preserved elements are solid in every design the tool returns.

**US-A3 (M):** As Dana, I want to mark keep-out regions, so that the part never collides with surrounding components.
- Keep-out elements are empty in every design the tool returns.

**US-A4 (M):** As Dana, I want to define supports and point or distributed loads, so that the optimizer designs for the real loading.

**US-A5 (S):** As Dana, I want to define several load cases with weights, so that the part is stiff under every way it will be loaded.

**US-A6 (M):** As Dana, I want to set the material budget and pick the material from a library of aerospace steels, aluminium alloys and titanium alloys, so that I control the part's weight and get the right stiffness and mass.

**US-A9 (S):** As Dana, I want to add a custom material by entering its properties, so that I can use an alloy or temper the library doesn't have.

**US-A7 (M):** As Dana, I want the tool to reject a setup that cannot be solved and tell me why, so that I don't wait for a run that will fail.
- Examples: no supports (rigid-body motion), a load inside a keep-out region, preserved regions alone exceeding the material budget.

**US-A8 (S):** As Dana, I want to save and load a problem setup as a file, so that I can rerun it and share it with Viktor.

### Epic B: Optimization

**US-B1 (M):** As Dana, I want to run the optimization and see the design evolve, so that I know it is progressing.

**US-B2 (M):** As Dana, I want to set a minimum member size, so that the part has no struts too thin to manufacture.

**US-B3 (M):** As Dana, I want the result to be a binary solid/empty design, so that it describes a real part rather than a cloud of grey density.

**US-B4 (S):** As Dana, I want to stop a run early and still get the best design found so far, with its certificate.

### Epic C: ML acceleration

**US-C1 (S):** As Dana, I want an instant predicted design when I finish the setup, so that I get feedback before the full optimization runs.

**US-C2 (S):** As Dana, I want the optimizer to start from the ML prediction, so that I get a certified result faster.
- When the ML prediction is poor, the result is still correct; only the speed is affected.

**US-C3 (C):** As Dana, I want several distinct candidate designs for the same setup, so that I can choose by criteria the optimizer doesn't know (looks, assembly, cost).

**US-C4 (C):** As Dana, I want fast approximate stiffness feedback while I edit the setup, so that I can explore what-ifs interactively.

**US-C5 (M):** As Mira, I want to generate a training dataset by running the physics optimizer on many random setups, so that models can be trained without outside data.

**US-C6 (M):** As Mira, I want a benchmark comparing optimization with and without ML warm start, so that I can show whether a model actually saves time at equal or better quality.

### Epic D: Verification and certificate

**US-D1 (M):** As Viktor, I want every returned design checked for feasibility, so that I know it respects the material budget, preserved regions and keep-out regions exactly.

**US-D2 (M):** As Viktor, I want a proven lower bound on the achievable compliance and the resulting optimality gap, so that I know how close the design is to the best possible.

**US-D3 (M):** As Viktor, I want the certificate to state the model's assumptions and the numerical tolerances, so that I know exactly what the proof covers.

**US-D4 (M):** As Viktor, I want to re-check a certificate with a standalone checker that shares no code with the optimizer or the ML, so that a bug in the optimizer cannot fake a proof.

**US-D5 (S):** As Viktor, I want a check of the first-order optimality (KKT) conditions for the optimizer's continuous solution, so that I know the optimizer converged to a local optimum.

**US-D6 (M):** As Viktor, I want the verifier tested against problems with known optimal solutions, so that I can trust the verifier itself.

**US-D7 (S):** As Viktor, I want a stress check at ultimate load with the margin of safety against the material's allowable stress, so that I see early whether the stiffest design is also strong enough.
- The report labels the stress check as a check, not as part of the certificate.

**US-D8 (M):** As Viktor, I want each design marked as meeting or missing the target optimality gap, so that I can tell at a glance whether it is good enough.

### Epic E: Results and export

**US-E1 (M):** As Dana, I want to see the final design with its mass, compliance and optimality gap, so that I can judge it at a glance.

**US-E2 (S):** As Dana, I want to export the part as smooth geometry (STL first; STEP later), so that I can take it into CAD and manufacturing.

**US-E3 (S):** As Viktor, I want to export the certificate as a machine-readable file plus a readable report, so that it can be archived with the design.

## 5. Requirements

### 5.1 Functional requirements

| ID | Requirement | Stories | Priority |
| --- | --- | --- | --- |
| FR-01 | The tool shall discretize a 2D rectangular design envelope into a structured mesh of square bilinear (Q4) elements at a user-set resolution. | A1 | M |
| FR-02 | The tool shall let the user assign each element to one of: design, preserved (fixed solid), keep-out (fixed empty). | A2, A3 | M |
| FR-03 | The tool shall let the user define supports (fixed degrees of freedom) and nodal or distributed loads. | A4 | M |
| FR-04 | The tool shall support multiple load cases combined as a weighted sum of compliances. | A5 | S |
| FR-05 | The tool shall take the material budget as a volume fraction and the material as linear, isotropic and elastic, defined by Young's modulus, Poisson's ratio, density and allowable stress. | A6 | M |
| FR-05a | The tool shall ship a material library with at least one titanium alloy (Ti-6Al-4V), two aluminium alloys (7075-T6, 2024-T3), one additively manufactured aluminium (AlSi10Mg) and one precipitation-hardened stainless steel (17-4PH). Each entry records the source of its values. | A6 | M |
| FR-05b | The tool shall let the user add custom materials with the same properties, saved with the problem setup. | A9 | S |
| FR-06 | The tool shall validate a setup before running and report each problem found: missing supports or rigid-body motion, loads inside keep-out regions, a budget smaller than the preserved volume, a budget below what any connecting design needs where detectable. | A7 | M |
| FR-07 | The tool shall save and load the complete problem setup in a versioned, human-readable file format. | A8 | S |
| FR-08 | The tool shall solve the minimum-compliance problem with a density-based method (SIMP) with a density filter enforcing a minimum member size. | B1, B2 | M |
| FR-09 | The tool shall report progress per iteration (iteration count, compliance, volume, change) and allow the run to be stopped. | B1, B4 | M |
| FR-10 | The tool shall convert the optimized density field into a binary design and keep only the binary design as the result. | B3 | M |
| FR-11 | The binary design shall be feasible: volume ≤ budget, preserved elements solid, keep-out elements empty, and a connected load path from every load to the supports. | B3, D1 | M |
| FR-12 | The tool shall provide an ML model that predicts a density field from a problem setup. | C1 | S |
| FR-13 | The optimizer shall accept a density field as its starting point and project it onto the feasible set before starting. | C2 | S |
| FR-14 | The ML model shall be able to produce several distinct candidates for one setup. | C3 | C |
| FR-15 | The tool shall provide a surrogate model that estimates compliance for a design in interactive time, and shall label its output as an estimate. | C4 | C |
| FR-16 | The tool shall include a dataset generator that samples random valid setups, runs the optimizer and stores setup, final design, compliance and certificate per sample. | C5 | M |
| FR-17 | The tool shall include a benchmark that runs a fixed set of problems with and without ML warm start and reports wall time, iterations, final compliance and optimality gap for each. | C6 | M |
| FR-18 | The verifier shall compute the compliance of the binary design (the upper bound) with a finite-element solve independent of the optimizer's own analysis. | D1, D2 | M |
| FR-19 | The verifier shall compute a lower bound LB from the convex relaxed problem, using a duality or Frank–Wolfe gap so that the value is a valid bound for any relaxed iterate, not only an exact optimum. | D2 | M |
| FR-20 | The verifier shall report the optimality gap `(c(x_final) − LB) / LB`. | D2, E1 | M |
| FR-20a | The verifier shall mark each design as meeting or missing the target optimality gap (section 6), with the target value shown. | D8 | M |
| FR-21 | The certificate shall state the mesh, material model (linear elasticity, small deformations), load cases, solver type, solver residual and the floating-point precision used. | D3 | M |
| FR-22 | A standalone certificate checker shall recompute feasibility, upper bound and lower bound from the certificate file alone, sharing no code with the optimizer or ML components. | D4 | M |
| FR-23 | The verifier shall report the KKT residual of the optimizer's continuous solution. | D5 | S |
| FR-23a | The tool shall compute von Mises stress in the binary design at ultimate load (limit load × 1.5 by default, user-adjustable) and report the minimum margin of safety against the material's allowable stress, with a fitting factor (default 1.15, per 14 CFR § 25.625) applied at preserved connection regions. The report shall state that this check is not part of the certificate. | D7 | S |
| FR-24 | The verifier shall have automated tests on reference problems with known optima: small problems solved exactly by enumeration or branch-and-bound, and analytical benchmarks (Michell-type structures) where applicable. | D6 | M |
| FR-25 | The tool shall display the final design with mass, compliance, lower bound and optimality gap. | E1 | M |
| FR-26 | The tool shall export the design boundary as smoothed geometry in STL. | E2 | S |
| FR-27 | The tool shall export the certificate as JSON and as a readable report. | E3 | S |

### 5.2 Non-functional requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| NFR-01 | **Soundness.** The tool shall never report a gap smaller than the true gap for the stated model, up to the stated numerical tolerance. When it cannot establish a bound, it shall say so instead of reporting a number. | M |
| NFR-02 | **ML is untrusted.** No result, metric or claim shown to the user shall depend on an ML output without verification. ML may only change speed and which candidates are proposed. | M |
| NFR-03 | **Reproducibility.** The same setup file, settings and random seed shall produce the same design and certificate. | M |
| NFR-04 | **Numerical transparency.** All certificate computations shall use IEEE 754 double precision. Linear solves shall reach a relative residual of at most 1e-10. Every certificate shall carry the achieved residual, and the gap shall be reported with its numerical tolerance taken into account. | M |
| NFR-04a | **Rigorous re-check.** The standalone checker should optionally recompute the bounds with interval arithmetic for meshes up to 100 × 50 elements, to show that floating-point error does not change the result. | C |
| NFR-05 | **Performance.** A 2D problem of 200 × 100 elements shall produce a certified result in under 60 s on the reference machine (AMD Ryzen 7 9800X3D) without ML. | S |
| NFR-06 | **ML benefit.** An ML warm start shall be kept only if the benchmark (FR-17) shows lower median wall time at an equal or better median optimality gap. | S |
| NFR-07 | **Extensibility to 3D.** The mesh, finite-element, optimizer and verifier interfaces shall not assume two dimensions, so that 3D can be added without redesigning them. | S |
| NFR-08 | **Testability.** The finite-element solver, optimizer, verifier and checker shall each be testable in isolation with automated tests. | M |
| NFR-09 | **Traceability.** Each certificate shall record the tool version and the ML model version (if one was used). | S |
| NFR-10 | **Platform.** The tool shall run as a desktop application on Windows 11. It shall be written in Python with NumPy/SciPy for the solver, PyTorch for ML and the usual Python ML libraries. | M |
| NFR-11 | **Hardware.** ML training and inference shall run on an NVIDIA RTX 3070 (8 GB VRAM) with CUDA. Every feature shall also work without a GPU, only slower. | M |
| NFR-12 | **Units.** All inputs and outputs shall carry SI units (mm, N, MPa, kg) and display them in the user interface and reports. | M |

### 5.3 Out of scope for the first version

- 3D design spaces (planned for a later version, see NFR-07).
- Stress constraints inside the optimization, fatigue, buckling, dynamics, thermal loads, plasticity and anisotropic (for example as-printed) material behaviour. The MVP only reports a stress check (FR-23a), and the certificate does not cover it.
- Airworthiness. A certified optimality gap says the design is efficient. It does not show the part is airworthy: that still needs the full stress analysis, material allowables (for example MMPDS) and testing.
- Proving global optimality over all possible meshes or for the physical part. The certificate covers the stated mesh and model only.
- Manufacturing constraints beyond minimum member size (overhang angles, milling directions, casting).

## 6. Target optimality gap

**Decision (proposed):** a certified gap of **≤ 5 %** is the acceptance target for the MVP, with **≤ 2 %** as the stretch goal. Every design reports its gap whether or not it meets the target.

**Rationale**

- **No industry standard exists.** Commercial tools (Altair OptiStruct, Autodesk Fusion, Ansys) stop when the objective changes by less than a relative tolerance between iterations. That shows the run has settled, not how far the design is from the best possible. Proving a gap would go beyond what commercial tools offer today, so the target is a project decision, not a compliance requirement.
- **Local methods can be far from optimal.** In certified global optimization studies of frame structures (Tyburec et al., 2021), local methods ended between 0 % and about 32 % above the proven global optimum, depending on the problem. Without a certificate, nobody knows which case a design is in.
- **What 5 % means in practice.** For stiffness-driven designs, compliance scales roughly with 1 / material volume. A 5 % gap therefore means about 5 % more mass than strictly needed for the same stiffness, on this mesh and model.
- **It is consistent with aerospace practice.** The design is still sized with a factor of safety of 1.5 on limit loads (14 CFR § 25.303) and a fitting factor of 1.15 at connections (§ 25.625). A ≤ 5 % efficiency gap is small next to these factors and next to the scatter in material allowables. Tighter than about 1–2 % is below what a linear-elastic finite-element model of a 2D mesh can meaningfully resolve.

**Main risk.** The simplest lower bound comes from the convex relaxation, where intermediate densities are allowed. Grey material is stiffness-efficient there, so this bound can sit well below the best solid-or-empty design. The reported gap may then exceed 5 % even when the design is close to optimal. Phase 1 must measure how large this relaxation gap is on the benchmark problems. If it is too large to show ≤ 5 %, the project needs a tighter lower bound (for example branch-and-bound on small sub-problems, or cutting planes). That is a research task, recorded as risk R1.

## 7. Numerical precision

**Decision:** double-precision floating point with a reported residual (NFR-04). Rigorous interval arithmetic is an optional re-check (NFR-04a).

Computers store numbers to about 16 significant digits and round after every operation. The finite-element solve is also iterative and stops at a small residual instead of the exact answer. The computed compliance and lower bound therefore carry a tiny numerical error, typically around 10⁻⁸ relative or smaller.

- **Floating-point certificate:** computed with these normal numbers. It reports the achieved residual, so a reader knows the size of the numerical error, but rounding is not formally accounted for.
- **Rigorous certificate:** computed with interval arithmetic, where every operation returns a range guaranteed to contain the true value. The bound then holds even counting every rounding error, but it is roughly 10–100× slower and harder to build.

For aerospace parts the floating-point version is enough. Its error is many orders of magnitude smaller than the model's assumptions (linear elasticity, the mesh, the loads) and the 1.5 factor of safety. Industrial finite-element analysis used in certification is likewise run in floating point. The optional interval re-check on small meshes is there to show this holds.

## 8. Risks

| ID | Risk | Mitigation |
| --- | --- | --- |
| R1 | The convex lower bound is too loose to show a ≤ 5 % gap. | Measure the relaxation gap in phase 1. Research tighter bounds if needed. |
| R2 | 8 GB VRAM limits model size and resolution for ML training. | Start at 128 × 64 resolution. Use resolution-free models (NITO-style) or mixed precision if needed. |
| R3 | The ML warm start brings no measurable speed-up. | The benchmark (FR-17) decides. The tool works without ML (NFR-02, NFR-06). |

## 9. Remaining open questions

1. **Target gap.** Confirm ≤ 5 % (MVP) and ≤ 2 % (stretch), or set other values.
2. **Desktop UI framework.** The proposal is PySide6 (Qt for Python), which is mature and works on Windows. Alternatives are Dear PyGui (simpler, GPU-drawn) and a local web interface wrapped as a desktop app.
3. **Material values.** The library values must come from a cited source (MMPDS, or the supplier's datasheet for AM materials). Do you have access to MMPDS, or should published typical values be used for now?

## 10. Sources

- 14 CFR § 25.303, Factor of safety: https://www.law.cornell.edu/cfr/text/14/25.303
- 14 CFR § 25.625, Fitting factors: https://www.law.cornell.edu/cfr/text/14/25.625
- Altair OptiStruct, DOPTPRM OBJTOL (relative convergence criterion): https://help.altair.com/hwsolvers/os/topics/solvers/os/doptprm_objtol_bulk_r.htm
- Tyburec, Zeman, Kružík, Henrion: Global optimality in minimum compliance topology optimization of frames and shells by moment-sum-of-squares hierarchy (2021): https://arxiv.org/abs/2009.12560
- Stolpe & Bendsøe: Truss topology optimization with discrete design variables, guaranteed global optimality and benchmark examples: https://orbit.dtu.dk/en/publications/truss-topology-optimization-with-discrete-design-variables-guaran/
- Woldseth, Aage, Bærentzen, Sigmund: On the use of artificial neural networks in topology optimisation (2022): https://arxiv.org/abs/2208.02563
- Sigmund: On benchmarking and good scientific practise in topology optimization (2022): https://link.springer.com/article/10.1007/s00158-022-03427-2
