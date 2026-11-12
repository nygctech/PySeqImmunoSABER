# PySeqImmunoSABER

Spatial proteomics on a repurposed Illumina HiSeq 2500. This repository holds the instrument
configuration, imaging recipe, and topic-modeling code for **"Spatial proteomics on repurposed
sequencers reveals early astrocytic remodeling in an ALS mouse model."**

## Summary

We repurposed an Illumina HiSeq 2500 as an automated fluorescence microscope with fluidics
control and ran a 28-antibody ImmunoSABER panel over 11 imaging cycles, decoding each cycle's
imagers automatically on the instrument. Applying the workflow to 70 lumbar spinal-cord sections
from SOD1-G93A and wild-type mice across three disease stages, we found that the environments
around motor neurons remodel at disease onset: GFAP-high neighborhoods expand through the
ventral gray matter, with **43% of motor neurons embedded in them in SOD1-G93A mice versus 15%
in wild type**. This reorganization precedes measurable motor-neuron loss, and the spatial
protein patterns agree with transcriptomic data from both mouse and human ALS tissue.

<img width="610" height="610" alt="Screenshot 2026-11-12 at 4 49 29 PM" src="https://github.com/user-attachments/assets/04fc8ee5-6851-4099-a069-c960ef7ff33e" />

## Links

| | |
|---|---|
| **Preprint** | [biorxiv.org/content/10.64898/2026.11.08.765207v1](https://www.biorxiv.org/content/10.64898/2026.11.08.765207v1) |
| **Data** | [10.5281/zenodo.23959523](https://doi.org/10.5281/zenodo.23959523) |
| **Sequencer repurposing** | [nygctech/PySeq2500](https://github.com/nygctech/PySeq2500) |
| **Image processing pipeline** | [nygctech/hudson](https://github.com/nygctech/hudson) |
| **Ecosystem** | [regenseq.github.io](https://regenseq.github.io/) |

## Workflow

| Stage | Runs with | Files here |
|---|---|---|
| 1. Acquisition — 11 cycles of hybridize / image / strip on the HiSeq 2500 | [PySeq2500](https://github.com/nygctech/PySeq2500) | [`pyseq2500/`](pyseq2500) |
| 2. Processing — unmixing, registration, segmentation, marker quantification | [hudson](https://github.com/nygctech/hudson) | [`hudson/`](hudson) |
| 3. Modeling — spatial topic models over the cell × marker matrix | `sceLDA` (this repo) | [`sceLDA/`](sceLDA) |

### `pyseq2500/` — acquisition

- `example_ImmunoSABER_config.txt` — experiment config: 11 cycles, two flowcells,
  coordinates for imaging regions of interests, imager-to-port assignments, laser powers,
  focus and z-stack settings.
- `ImmunoSABER_recipe.txt` — fluidics recipes: the per-cycle imager exchange (formamide strip →
  PBS → imager hybridization at 37 °C → image), and the pre-recipe that hybridizes concatemers
  and images the tissue background.

Run with `pyseq -c example_ImmunoSABER_config.txt`; see the
[PySeq2500 docs](https://github.com/nygctech/PySeq2500) for installation and the paths in the
`[ImmunoSABER]` section, which point at the machine's local recipe files.

### `hudson/` — image processing

`hudson_config.yaml` maps each imaging cycle and emission channel to its marker (e.g. cycle 7 /
558 nm → GFAP), and configures spectral unmixing, Cellpose segmentation on the DRAQ5 (nuclei)
and polyT (cytoplasm) channels, feature extraction, and Dask/SLURM resources. Point
[hudson](https://github.com/krpandit/hudson) at it to turn a raw PySeq run into a cell ×
marker matrix.

### `sceLDA/` — spatial topic modeling

Semantic segmentation of cellular communities from high-dimensional spatially-resolved data —
a spatial latent Dirichlet allocation (SLDA) model that assigns cells to spatially-coherent
"documents" and learns topics over marker vocabularies, optionally on a VAE embedding. Used
here to define the neighborhoods (including the GFAP-high ones) the paper reports.

## Installing sceLDA

Python ≥ 3.8. Clone with git rather than downloading a zip — the version is derived from git
tags by `setuptools_scm`, so a source download without `.git` fails to build.

```bash
git clone https://github.com/nygctech/PySeqImmunoSABER.git
cd PySeqImmunoSABER

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

pip install -e ./sceLDA
```

Check the install:

```bash
python -c "from scelda import slda, image_processing; print(slda.run_slda.__doc__ or 'ok')"
```

`torch` installs as the default CPU/GPU wheel for your platform. For a specific CUDA build,
install `torch` from the [PyTorch index](https://pytorch.org/get-started/locally/) *before*
`pip install -e ./sceLDA`.

## Using sceLDA

The package exposes a Python API; there is no CLI.

```python
import numpy as np
from scelda import slda, image_processing, spatialdata_preprocessing

# cluster per-cell marker vectors into a discrete "vocabulary"
assignments, kmeans = image_processing.cluster_descriptors(all_words, vocab_size=100)

# spatial coordinates -> points array
points = spatialdata_preprocessing.make_points_arr(loc_df, 'x', 'y')

# fit the spatial topic model
pi_hat, phi_hat, best_topic = slda.run_slda(
    assignments, points,
    n_topics=5, vocab_size=100, n_docs=200, n_imgs=1,
    sigma=250, n_epochs=30,
)

slda.plot_slda_results(best_topic, 'topics.png', points)
```

Importable modules: `slda` (spatial LDA: initialization, Gibbs training, topic assignment,
plotting), `image_processing` (descriptor extraction and clustering), `spatialdata_preprocessing`
(coordinate and neighborhood helpers), `model` / `VAE` / `utils` (VAE and early stopping),
`evaluate`, `histogram_clustering`, `simulate_classes`.

### Notebooks

`sceLDA/notebooks/` holds the worked examples — `RunSLDA.ipynb` (SLDA end to end),
`sceLDA_slideseq.ipynb`, `VAE.ipynb`, `image_processing.ipynb`,
`spatial_data_preprocessing.ipynb`, `simulated_data.ipynb`.

They import the modules by bare name (`import slda`), so put the package source on the path
first:

```python
import sys; sys.path.append('../src/scelda')
```

or launch Jupyter from `sceLDA/src/scelda`.

### Caveats

A few rough edges in the research code, left as-is for reproducibility:

- **Bare intra-package imports.** `train_slda.py`, `train_vae.py`, `featurize_samples.py`,
  `train_sceLDA_endtoend.py`, and the body of `sceLDA.train_scelda()` use `import slda` /
  `from model import VAE` rather than relative imports, so they only resolve when
  `sceLDA/src/scelda` is on `sys.path`. The modules listed under *Using sceLDA* above import
  cleanly as `scelda.*`.

## Data

Processed data are on Zenodo at [10.5281/zenodo.23959523](https://doi.org/10.5281/zenodo.23959523)
(CC-BY 4.0):

- `all_05012026.h5ad` (6.9 GB) — AnnData with per-cell marker intensities, spatial coordinates,
  and section metadata for all 70 sections.
- `example_ImmunoSABER_config.txt`, `ImmunoSABER_recipe.txt`, `hudson_config.yaml` — copies of
  the configs in this repo.

`sceLDA/data/` additionally holds the small Slide-seq pucks and MSRC-v2 images used by the
example notebooks.

## Citation

> Pandit K, Fouts C, Rodwin S, Maniatis S, Dhingra K, Marrakchi S, Domovic D, Singh J,
> Dumitrascu B, Phatnani H, Pleska M, Vickovic S. *Spatial proteomics on repurposed sequencers
> reveals early astrocytic remodeling in an ALS mouse model.* bioRxiv (2026).
> doi:10.64898/2026.11.08.765207
