# Generative Design Tool

2D topology optimization for aerospace parts (titanium, aluminium, steel) with a **certified optimality gap**: every result comes with a proof of how far, at most, its stiffness is from the best possible design on the same mesh, plus a standalone checker that re-verifies that proof.

Requirements and decisions: [`docs/requirements.md`](docs/requirements.md). Material data: [`data/materials.json`](data/materials.json).

## Install

Python 3.11 or newer.

```
pip install -e engine -e checker
```

## Run

```
gdt examples/bracket.json --out runs/bracket
gdt-check runs/bracket/certificate.json
```

`gdt` validates the setup, optimizes, and writes to the output folder:

| File | Content |
| --- | --- |
| `certificate.json` | Setup, material, binary design, bounds, gap, numerics, stress check |
| `report.md` | Readable summary with the image, gap, target status and stress check |
| `design.png` | The design with preserved regions, keep-out regions, supports and loads |
| `design_grid.txt` | One line per element row (top row first), `1` = material |

`gdt-check` reads only the certificate. It shares no code with the engine, and it exits with code 0 only if every claim is confirmed.

## Setup file

JSON, units mm / N / MPa. Origin bottom-left, y up. See [`examples/bracket.json`](examples/bracket.json).

| Field | Meaning |
| --- | --- |
| `envelope` | `width_mm`, `height_mm`, `thickness_mm` |
| `mesh` | `nx`, `ny` (elements must be square) |
| `material` | `{"id": "ti-6al-4v"}` from the library, optional `"allowable": {"product_form": "sheet", "basis": "B"}`; or `{"custom": {"name", "youngs_modulus_GPa", "poissons_ratio", "density_kg_m3", "yield_MPa"?, "ultimate_MPa"?}}` |
| `material_budget` | Maximum volume fraction (0, 1] |
| `min_member_size_mm` | Filter radius of the optimizer |
| `preserved_regions` | Rectangles that must stay solid (connections) |
| `keep_out_regions` | Rectangles or circles that must stay empty |
| `supports` | Rectangles (zero width/height for lines or points); `fix`: `x`, `y` or `xy` |
| `load_cases` | `name`, `weight`, `loads` (limit loads): `point` (`x_mm`, `y_mm`, `fx_N`, `fy_N`) or `edge` (`edge`, `from_mm`, `to_mm`, total `fx_N`, `fy_N`) |
| `settings` | Optional: `max_iterations`, `tolerance`, `factor_of_safety` (1.5), `fitting_factor` (1.15) |

## Tests

```
pip install pytest mypy
pytest            # fast suite
pytest -m slow    # 200 x 100 performance benchmark
mypy
```
