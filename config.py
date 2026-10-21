from pathlib import Path

CSCS_ROOT = Path("/capstor/scratch/cscs/sbrunne")
CSCS_THESIS = CSCS_ROOT / "Thesis"

UZH_ROOT = Path("/shares/feldmann.ics.mnf.uzh/Santiago")
UZH_THESIS = UZH_ROOT / "Thesis"

DEFAULT_SNAPSHOTS = (60, 63, 66, 69, 72, 75, 80, 85, 90, 95, 100)

# Inverse WDM particle mass (1/keV) per dark matter model. CDM is treated
# as the m_WDM -> infinity limit, hence inverse mass 0.
INV_MASSES = {
    "wdm03d5": 1 / 3.5,
    "wdm05": 1 / 5.0,
    "wdm07d5": 1 / 7.5,
    "wdm10": 1 / 10.0,
    "cdm": 0.0,
}

EXCLUDED_PAIRS: tuple[tuple[int, str], ...] = (
    (15, "wdm03d5"), (25, "wdm03d5"), (32, "wdm03d5"), (34, "wdm03d5"),
    (38, "wdm03d5"), (42, "wdm03d5"), (48, "wdm03d5"), (49, "wdm03d5"),
    (51, "wdm03d5"),
    (20, "wdm05"), (32, "wdm05"), (42, "wdm05"), (46, "wdm05"),
    (48, "wdm05"), (51, "wdm05"),
)

DM_MODELS_3 = ["wdm07d5", "wdm10", "cdm"]
DM_MODELS_5 = ["wdm03d5", "wdm05", "wdm07d5", "wdm10", "cdm"]

ROT_TAG_MAP = {"random180": "", "rot90": "_rot90", "none": "_norot"}
