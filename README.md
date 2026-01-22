# playNano Plugins

A collection of **optional analysis modules, processing filters, and plotting helpers**
for use with the [playNano](https://github.com/derollins/playNano) platform.

This repository contains extensions that are useful alongside playNano but are not
part of the core package.

---

## Background

`playNano` provides a modular framework for processing and analysing time series AFM
image stacks and videos. This plugin repository extends that framework with:

- custom **analysis modules**
- optional **processing filters** (including third-party workflows)
- lightweight **plotting utilities** for inspecting analysis results

Keeping these components separate allows optional dependencies and faster iteration
on experimental or domain-specific workflows.

The full playNano documentation can be found here: <https://derollins.github.io/playNano>

---

## Contents

### Analysis plugins

- **`particle_boundary_size`**
  Measures the maximum bounding-box dimension of tracked particles over time, with
  optional threshold-based state classification.

### Processing filters

- **`topostats_filter`** *(optional dependency)*
  A wrapper around the TopoStats filtering pipeline, exposed in a form suitable for
  use from the playNano CLI and API.

### Plotting helpers

- Static and animated plotting functions for boundary-size analysis, including
  single-track plots, multi-track overlays, and cropped mask animations.

---

## Installation

### Clone the repository from GitHub

```bash
git clone https://github.com/derollins/playNano-plugins
cd playNano-plugins
```

### Basic install (no optional dependencies)

```bash
pip install -e .
```

or from GitHub:

```bash
pip install git+https://github.com/derollins/playNano-plugins
```

With optional TopoStats support:

```bash
pip install -e ".[topostats]"
```

TopoStats is only required if you use the topostats_filter processing plugin.

## Usage

### Using analysis plugins in a playNano pipeline

Plugins are discovered automatically via entry points once installed.

```python
from playnano.analysis.pipeline import AnalysisPipeline

p = AnalysisPipeline()
p.add("feature_detection", mask_fn="mask_mean_offset", factor=3)
p.add("particle_tracking", max_distance=80)
p.add("particle_boundary_size", threshold=66)

record = p.run(stack)

out = record["analysis"]["step_3_particle_boundary_size"]
```

The analysis output includes:

- `per_track` – per-particle time series
- `flat_table` – row-wise records suitable for DataFrames
- `summary` – basic bookkeeping information
- `plot_hints` – suggested plotting helpers

### Plotting results

```python
from playnano_plugins.plotting import plot_boundary_over_time

fig = plot_boundary_over_time(out, track_id=0)
fig.show()
```

Animated sanity-check visualisation:

```python
from playnano_plugins.plotting import animate_boundary_size_crop

fd = record["analysis"]["step_1_feature_detection"]

fig, anim = animate_boundary_size_crop(
    boundary_out=out,
    masks=fd["labeled_masks"],
    track_id=0,
)
```

### Using the TopoStats filter

This filter uses the [TopoStats](https://github.com/AFM-SPM/TopoStats) filter module to
filter and flatten AFM image frames.

See the TopoStats Flattening documentation for more details on usage:
[https://afm-spm.github.io/TopoStats/main/advanced/flattening.html](https://afm-spm.github.io/TopoStats/main/advanced/flattening.html)

```python
from playnano_plugins.processing import topostats_filter

filtered = topostats_filter(
    frame,
    gaussian_size=1.0,
    threshold_method="std_dev",
    remove_scars=True,
)
```

Most TopoStats parameters can be configured directly via function arguments or by
passing a configuration dictionary.

---

## Writing your own plugins

For guidance on writing custom analysis modules that integrate with playNano, see:

[https://derollins.github.io/playNano/main/custom_analysis_modules.html](https://derollins.github.io/playNano/main/custom_analysis_modules.html)

That guide covers:

- the analysis module base class
- required method signatures
- provenance handling
- registration and discovery via entry points

## License

This project is licensed under the GNU GPL v3 or later, consistent with playNano
and compatible third-party dependencies.
