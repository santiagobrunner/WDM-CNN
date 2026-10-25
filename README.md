# WDM-CNN: Distinguishing dark matter models from deep galaxy imaging

Code for classifying cold (CDM) and warm (WDM) dark matter models from mock deep
images of Milky-Way-mass galaxies, and for measuring how deep a survey must
observe to tell them apart.

> Accompanies: *TODO: paper title, authors, arXiv/DOI link.*
> Developed from the Master's thesis "Distinguishing Dark Matter Models From Deep
> Galaxy Imaging" (S. Brunner, UZH / ETH Zurich, 2026).

## Overview

Cosmological zoom-in simulations (FIREbox) of 15 Milky-Way-mass halos are run with
identical initial conditions but three dark matter models: CDM, WDM 10 keV and
WDM 7.5 keV. Star particles are projected along 24 lines of sight at 11 redshifts
(z = 0.5 to 0) into mock images for the different Euclid-bands (VIS, Y, J, H), and Gaussian noise
is added to reach a chosen surface-brightness limit (SBlim). A convolutional neural
network is then trained to classify the dark matter model, and its accuracy is
measured as a function of SBlim.

```
FIREbox zooms -> extract particles -> pNbody files -> noiseless images (24 LOS)
   -> add noise (SBlim) -> [mask satellites] -> train CNN -> evaluate
```

Main result (from the thesis): accuracy rises from chance (33 %) to a plateau
of about 76 % with a transition at SBlim of about 32-33 mag/arcsec^2. CDM is
separable from WDM (binary accuracy above 96 % at SBlim 35); 7.5 keV and 10 keV
WDM are not separable. TODO: update to the final paper numbers.

## Repository structure

```
src/wdm_cnn/       the package
    data/            loading images, labels, DataLoaders
    preprocessing/   pixel scalers, multi-band input
    models/          SimpleCNN, ConfigCNN, datasets
    imaging/         projection and satellite masking
    simulation/      AHF halo catalogues
    physics/         scaling relations (stellar mass -> halo size)
    evaluation/      inference, N-image aggregation, calibration, metrics, Grad-CAM
scripts/           entry points for each pipeline step (see below)
slurm/             SLURM job scripts
instruments/       Euclid VIS / Y / J / H instrument definitions for pNbody
splits/            train / validation / test halo splits (v1 ... v10)
hp_nside2_centers.txt   the 24 lines of sight
config.py          paths and constants (TODO: move into the package)
tests/             pytest unit tests
```

## Installation

```bash
git clone https://github.com/santiagobrunner/WDM-CNN.git
cd WDM-CNN
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Dependencies are listed in `pyproject.toml`. Image generation also needs
[pNbody](https://gitlab.com/obswww/pNbody), which is not on PyPI and must be
installed separately. Training and evaluation do not
need pNbody except for the noise function.

On the UZH cluster everything runs inside an Apptainer container
(`conda.sif`); the SLURM scripts in `slurm/` show how it is called.
TODO: document the container and how to build it.

## Data

TODO: the FIREbox zoom simulations are not public. Describe how to obtain them or
who to contact, and where the generated images (`input_images_<SBlim>_f/`) are
stored. Small files needed to rerun the pipeline are in this repository
(`splits/`, `instruments/`, `hp_nside2_centers.txt`).

## Usage

Each step below reads the output of the previous one. Paths are set in
`config.py` or passed as arguments.

| Step | Script | Notes |
|---|---|---|
| 1. Extract particles of a halo | `scripts/extract_region_value_added.py` | from a GIZMO snapshot |
| 2. Convert to pNbody format | `scripts/halo2pnb.py` | |
| 3. Noiseless mock images | `scripts/mockimgs_sb_compute_images`, `slurm/create_input_images.slurm` | uses `instruments/` and `hp_nside2_centers.txt` |
| 4. Add noise for a given SBlim | `slurm/add_noise_to_images.slurm` | |
| 5. Find the main halo in each AHF file | `scripts/find_main_halo.py` | see below |
| 6. Mask satellite galaxies (optional) | TODO: `scripts/make_masked_images.py` | |
| 7. Train a CNN | `scripts/train_cnn.py` | |
| 8. Random architecture configs (optional) | `scripts/generate_search_configs.py` | |
| 9. Evaluate | `wdm_cnn.evaluation` | TODO: analysis scripts |

### Finding the main halo

The AHF catalogues are sorted by mass, but in some zooms two halos have similar
mass (e.g. due to mergers), so the main halo is not always the first row. `find_main_halo.py` decides it
with a fixed cascade (mass, velocity, distance to the box centre at z = 0; then
position tracking to earlier snapshots), writes a CSV of halo indices and flags
every uncertain case.

```bash
python scripts/find_main_halo.py --overrides resources/halo_index_overrides.csv \
    -o halo_index_results.csv
```

Flags other than `OK` need a manual check.

### Training

```bash
python scripts/train_cnn.py --scaler shiftedlog --SB_Limit 35 --snapshots all \
    --n_classes 3 --augment --rotation rot90 --split_version v1
```

TODO: add the SLURM wrapper and the table of paper figures:

| Figure / table | Script | Input |
|---|---|---|
| TODO | TODO | TODO |

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

TODO: BibTeX for the paper. A `CITATION.cff` file lets GitHub show a "Cite this
repository" button.

## License and credits

See `LICENSE`. `extract_region_value_added.py` and `halo2pnb.py` are adapted from
code by R. Feldmann and L. Tortora, and `mockimgs_sb_compute_images` is adapted
from pNbody.

Built on the FIREbox simulations, the pNbody package, the AHF halo finder and
BPASS stellar population models.

## Contact

Santiago Brunner, santiago.brunner@yahoo.com