# import pandas as pd
# import numpy as np
# from astropy.io import fits
# from pNbody import Nbody
# from wdm_cnn.evaluation.metrics import get_summary
# from pathlib import Path
# from config import UZH_ROOT, UZH_THESIS, DEFAULT_SNAPSHOTS
# from wdm_cnn.data.loaders import dm_models_for

# input_path = UZH_ROOT / "input_images"
# dm_models = dm_models_for(5)
# snapshots = DEFAULT_SNAPSHOTS
# ages = ['474', '405', '342', '284', '231', '187', '144', '104', '066', '031', '000']

# halos = np.sort((input_path / "100").iterdir())
# print(f"{len(halos)} Halos found: {halos}")

#!/usr/bin/env python3
"""Find the main halo (row index in the AHF file) for every zoom run and snapshot.

The AHF files are sorted by mass, but in some zooms two halos have nearly the
same mass, so the main halo is not always row 0. This script decides it with a
fixed cascade of criteria (no randomness, no hand-edited state). The first
criterion that gives a clear answer decides; the others act as cross-checks.

z=0 (anchor snapshot), reference = the halo in the parent FIREbox catalogue:
    mass  ->  velocity  ->  distance to the box centre (7500, 7500, 7500)
Earlier snapshots, reference = the halo chosen at the next later snapshot of
the same zoom (the parent file only exists at z=0):
    position (distance to the reference halo)  ->  velocity
Decisions made by inspecting images live in an overrides CSV
(halo_nr, dm, snap, haloIndex, reason). They are applied while tracking, so
they carry over to the earlier snapshots.

Flags (anything that is not OK needs a look):
    OK               one criterion was clear and no other criterion disagrees
    CONFLICT         a clear criterion picked a different halo than the one chosen
    NEEDS_CHECK      no criterion was clear -> check by plotting the candidates
    FAR              tracking: chosen halo is far from the reference position
    MASS_JUMP        chosen halo's mass differs a lot from the reference mass
    MANUAL_OVERRIDE  taken from the overrides CSV

Output columns: halo_nr, dm, snap, haloIndex, haloID, flag, method, mass_dev,
dist_ckpch, dist_ratio, auto_haloIndex
  haloIndex  row of the main halo in the AHF file (0-based)
  haloID     value of the ID column in that row (the masking code compares
             this with haloIndex, so they must be equal)
  method     criterion that decided: mass, velocity, centre, position, none
  mass_dev   |M / M_ref - 1|; M_ref = parent mass at z=0, otherwise the mass
             of the halo chosen at the next later snapshot
  dist_ckpch z=0: distance to the box centre; otherwise: distance to the
             reference halo [ckpc/h]
  dist_ratio 2nd-best / best of the deciding criterion (large = clear)

Usage:
    python find_main_halo.py --overrides resources/halo_index_overrides.csv \
        --compare halo_index_results.csv -o halo_index_new.csv
"""

import argparse
import glob
import os
from config import DEFAULT_SNAPSHOTS
from wdm_cnn.data.loaders import dm_models_for
import numpy as np
import pandas as pd

BOX = 15000.0                      # FIREbox side length [ckpc/h]
CENTRE = np.full(3, BOX / 2)       # the zoom target region sits at the box centre
N_CAND = 3                         # look at the 3 most massive halos of each file
# A criterion is "clear" if (2nd-best - best) / 2nd-best >= GAP[criterion].
# Starting values; calibrate with dist_ratio of cases you checked by eye.
GAP = {"mass": 0.15, "velocity": 0.15, "centre": 0.15, "position": 0.67}
TRACK_FAR = 150.0                  # tracking: farther than this [ckpc/h] -> FAR
MASS_TOL = 0.5                     # |M/M_ref - 1| above this -> MASS_JUMP

POS = ["Xc", "Yc", "Zc"]
VEL = ["VXc", "VYc", "VZc"]
DM_MODELS = dm_models_for(5)




def read_ahf(path):
    """Read an AHF_halos file; column names without the '(n)' suffix."""
    with open(path) as f:
        names = [n.split("(")[0] for n in f.readline().lstrip("#").split()]
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=names)
    return df.rename(columns={"Mhalo": "Mvir"})  # zoom files say Mhalo, parent Mvir


def periodic_dist(a, b):
    """Distance in the periodic box (minimum image)."""
    d = np.abs(a - b)
    d = np.minimum(d, BOX - d)
    return np.linalg.norm(d, axis=-1)


def best_and_ratio(values):
    """Index of the smallest value and (2nd smallest / smallest)."""
    order = np.argsort(values, kind="stable")  # ties -> lowest row, deterministic
    best = int(order[0])
    if len(values) < 2:
        return best, np.inf
    first, second = values[best], values[order[1]]
    if second == first:      # exact tie (e.g. both 0): not clear
        return best, 1.0
    return best, float(second / first) if first > 0 else np.inf


def is_clear(name, ratio):
    return ratio >= 1 / (1 - GAP[name])


def decide(criteria):
    """criteria: ordered list of (name, values); smaller value = better match.

    The first clear criterion decides. Any other clear criterion that picks a
    different halo makes the result a CONFLICT. If none is clear -> NEEDS_CHECK
    (the best of the first criterion is returned).
    Returns (index, method, ratio, flag).
    """
    results = [(name, *best_and_ratio(v)) for name, v in criteria]
    clear = [(n, b, r) for n, b, r in results if is_clear(n, r)]
    if not clear:
        name, best, ratio = results[0]
        return best, "none", ratio, "NEEDS_CHECK"
    name, best, ratio = clear[0]
    flag = "CONFLICT" if any(b != best for _, b, _ in clear) else "OK"
    return best, name, ratio, flag


def load_overrides(path):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.duplicated(["halo_nr", "dm", "snap"]).any():
        raise ValueError(f"duplicate keys in {path}")
    return {(int(r.halo_nr), r.dm, int(r.snap)): int(r.haloIndex) for r in df.itertuples()}


def halo_number(halo):
    return int(halo.split("_")[1][1:])  # 'FB15_h32_L11_r4_p8' -> 32


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-path", default="/shares/feldmann.ics.mnf.uzh")
    ap.add_argument("--parent-ahf", default=None,
                    help="parent FIREbox z=0 AHF file (default: FB15N1024_old, snapshot 1200)")
    ap.add_argument("--halos", nargs="+", default=None,
                    help="full halo directory names; default: all FB15_h*_L11_* found")
    ap.add_argument("--dms", nargs="+", default=DM_MODELS)
    ap.add_argument("--snaps", nargs="+", type=int, default=DEFAULT_SNAPSHOTS)
    ap.add_argument("--overrides", default="resources/halo_index_overrides.csv")
    ap.add_argument("--compare", default=None, help="old results CSV to diff against")
    ap.add_argument("-o", "--output", default="halo_index_results.csv")
    args = ap.parse_args()

    base = args.base_path
    analysis = f"{base}/data/FIREbox/zooms/analysis"
    parent_path = args.parent_ahf or (f"{base}/data/FIREbox/analysis/AHF/FB15N1024_old/"
                                      "halo/1200/FB15N1024.z0.000.AHF_halos")
    halos = args.halos or [os.path.basename(p) for p in glob.glob(f"{analysis}/FB15_h*_L11_*")]
    halos = sorted(halos, key=halo_number)
    snaps = sorted(args.snaps, reverse=True)  # z=0 first, then backwards in time
    print(f"halos: {[halo_number(h) for h in halos]}")

    parent = read_ahf(parent_path)
    overrides = load_overrides(args.overrides)
    rows, missing = [], []

    for halo in halos:
        nr = halo_number(halo)
        prow = parent.iloc[nr]  # parent catalogue row nr is halo nr
        if prow["ID"] != nr:
            print(f"WARNING: parent row {nr} has ID {prow['ID']}")
        for dm in args.dms:
            ref_mass, ref_vel, ref_pos = prow["Mvir"], prow[VEL].to_numpy(float), None

            for snap in snaps:
                files = glob.glob(f"{analysis}/{halo}/{dm}/halo/{snap:03d}/*.AHF_halos")
                if len(files) != 1:
                    missing.append((nr, dm, snap, len(files)))
                    if ref_pos is None:
                        break  # no z=0 file -> nothing to anchor on
                    continue
                df = read_ahf(files[0])
                pos = df[POS].to_numpy(float)
                cand = slice(0, N_CAND)
                mass_dev = np.abs(df["Mvir"].to_numpy(float)[cand] / ref_mass - 1)
                vel_diff = np.linalg.norm(df[VEL].to_numpy(float)[cand] - ref_vel, axis=1)

                if ref_pos is None:  # z=0 anchor: mass -> velocity -> box centre
                    ref_dist = periodic_dist(pos, CENTRE)
                    criteria = [("mass", mass_dev), ("velocity", vel_diff),
                                ("centre", ref_dist[cand])]
                else:                # tracking: position -> velocity
                    ref_dist = periodic_dist(pos, ref_pos)
                    criteria = [("position", ref_dist[cand]), ("velocity", vel_diff)]
                idx, method, ratio, flag = decide(criteria)
                auto_idx = idx

                key = (nr, dm, snap)
                if key in overrides:
                    idx, method, flag = overrides[key], "override", "MANUAL_OVERRIDE"
                dist = float(ref_dist[idx])
                mdev = float(abs(df["Mvir"].iloc[idx] / ref_mass - 1))
                if flag == "OK" and ref_pos is not None and dist > TRACK_FAR:
                    flag = "FAR"
                if flag == "OK" and mdev > MASS_TOL:
                    flag = "MASS_JUMP"

                rows.append(dict(halo_nr=nr, dm=dm, snap=snap, haloIndex=idx,
                                 haloID=int(df["ID"].iloc[idx]), flag=flag, method=method,
                                 mass_dev=mdev, dist_ckpch=dist, dist_ratio=ratio,
                                 auto_haloIndex=auto_idx))
                # the chosen halo is the reference one snapshot earlier
                ref_pos = pos[idx]
                ref_mass = df["Mvir"].iloc[idx]
                ref_vel = df[VEL].iloc[idx].to_numpy(float)

    out = pd.DataFrame(rows).sort_values(["halo_nr", "dm", "snap"]).reset_index(drop=True)
    out.to_csv(args.output, index=False, float_format="%.6g")

    print(f"\nwrote {len(out)} rows to {args.output}")
    print("decided by:", out["method"].value_counts().to_dict())
    if missing:
        print(f"missing/ambiguous AHF files ({len(missing)}): {missing}")
    bad = out[out["flag"] != "OK"]
    print(f"rows needing a look: {len(bad)}/{len(out)}")
    print(bad[["halo_nr", "dm", "snap", "haloIndex", "flag", "method", "dist_ratio"]]
          .to_string(index=False))
    n_id = int((out["haloID"] != out["haloIndex"]).sum())
    if n_id:
        print(f"\nWARNING: haloID != haloIndex in {n_id} rows. The masking code uses "
              "haloIndex as an AHF ID, so check that.")
    ov = out[(out["flag"] == "MANUAL_OVERRIDE") & (out["haloIndex"] != out["auto_haloIndex"])]
    print(f"\noverrides that differ from the automatic choice: {len(ov)} "
          f"(of {int((out['flag'] == 'MANUAL_OVERRIDE').sum())})")

    if args.compare:
        old = pd.read_csv(args.compare)
        m = out.merge(old, on=["halo_nr", "dm", "snap"], suffixes=("", "_old"))
        diff = m[m["haloIndex"] != m["haloIndex_old"]]
        print(f"\nvs {args.compare}: {len(diff)} of {len(m)} rows differ")
        print(diff[["halo_nr", "dm", "snap", "haloIndex", "haloIndex_old", "flag"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()