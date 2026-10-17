
import numpy as np
import pandas as pd
import glob



# -----------------------------------------------------------------
# exact pNbody rotation convention (obs_object.py / utils.py::align)
# -----------------------------------------------------------------
def resolve_ahf_path(base_path, halo, dm, snap):
    pattern = f"{base_path}/data/FIREbox/zooms/analysis/{halo}/{dm}/halo/{snap:03d}/*.AHF_halos"
    matches = glob.glob(pattern)
    if len(matches) == 0:
        raise FileNotFoundError(f"no AHF_halos file found for pattern:\n  {pattern}")
    if len(matches) > 1:
        raise ValueError(f"multiple matches:\n  " + "\n  ".join(matches))
    return matches[0]


def resolve_host_id(lookup_df, halo_nr, dm, snap):
    """
    lookup_df : dataframe with columns halo_nr, dm, snap, haloIndex, flag, mass_dev
                haloIndex == the AHF ID(1) value of the main halo for that
                (halo_nr, dm, snap) triple -- this IS what AHF's hostHalo
                column references (AHF convention: hostHalo stores the
                parent's ID(1), not a row position), so it can be used
                directly as host_id.
    """
    row = lookup_df[(lookup_df["halo_nr"] == halo_nr) &
                     (lookup_df["dm"] == dm) &
                     (lookup_df["snap"] == snap)]
    if len(row) != 1:
        raise ValueError(f"expected exactly 1 match for halo_nr={halo_nr}, "
                          f"dm={dm}, snap={snap}, got {len(row)}")
    return int(row["haloIndex"].iloc[0])


def read_ahf_satellites(ahf_file, host_id, hubble_h=0.6774, n_star_min=7, ):
    """
    Minimal AHF _halos reader for satellite subhalo centers.
    Handles the '#'-commented header manually (pd.read_csv chokes on it
    otherwise -- see project notes on the recurring AHF header bug).

    host_id : the AHF ID(1) value of the target (central) halo -- e.g.
              from resolve_host_id() / your halo_nr,dm,snap,haloIndex
              lookup table. Selects rows with hostHalo == host_id (NOT
              hostHalo == -1, which is the field-halo flag).
    """
    import pandas as pd

    with open(ahf_file) as f:
        header_line = f.readline()
    names = header_line.lstrip("#").split()
    names = [n.split("(")[0] for n in names]  # strip "(N)" suffixes

    df = pd.read_csv(ahf_file, sep=r"\s+", comment=None, skiprows=1,
                      names=names, header=None)
    
    central_row = df[df["ID"] == host_id]
    if len(central_row) != 1:
        raise ValueError(f"expected exactly 1 halo with ID={host_id} in "
                          f"{ahf_file}, got {len(central_row)}")
    if central_row["hostHalo"].iloc[0] != -1:
        print(f"WARNING: host_id={host_id} has hostHalo="
              f"{central_row['hostHalo'].iloc[0]} (not -1) -- this may not "
              f"be a true central/field halo. Check host_id resolution "
              f"before trusting the satellite selection below.")
        
    sats = df[(df["hostHalo"] == host_id) & (df["n_star"] >= n_star_min)]

    # AHF Xc/Yc/Zc are in kpc/h
    pos = sats[["Xc", "Yc", "Zc"]].to_numpy() / hubble_h
    mstar = sats["M_star"].to_numpy() / hubble_h if "M_star" in sats else None

    return pos, mstar


def read_satellites_by_radius(ahf_file, host_id, radius_ckpc=237.09, hubble_h=0.6774, n_star_min=10, min_dist_kpc=1):
    """
    Alternative to hostHalo-based selection: all halos (any hostHalo value,
    except the host itself) within radius_kpc of the host center, regardless
    of AHF's own structure classification.
    """
    radius_kpc = radius_ckpc / hubble_h  # AHF Xc/Yc/Zc are in kpc/h

    with open(ahf_file) as f:
        header_line = f.readline()
    names = header_line.lstrip("#").split()
    names = [n.split("(")[0] for n in names]  # strip "(N)" suffixes
    df = pd.read_csv(ahf_file, sep=r"\s+", skiprows=1, names=names, header=None)

    central_row = df[df["ID"] == host_id]

    if len(central_row) != 1:
        raise ValueError(f"expected exactly 1 halo with ID={host_id}, got {len(central_row)}")
    
    xc, yc, zc = central_row[["Xc", "Yc", "Zc"]].to_numpy()[0] / hubble_h

    pos_all = df[["Xc", "Yc", "Zc"]].to_numpy() / hubble_h
    dist = np.linalg.norm(pos_all - np.array([xc, yc, zc]), axis=1)

    sel = (df["ID"] != host_id) & (dist <= radius_kpc) & (df["n_star"] >= n_star_min) & (dist >= min_dist_kpc)
    sats = df[sel]

    pos = sats[["Xc", "Yc", "Zc"]].to_numpy() / hubble_h
    mstar = sats["M_star"].to_numpy() / hubble_h if "M_star" in sats else None
    return pos, mstar


def get_halo_center(ahf_file, host_id, hubble_h=0.6774):
    import pandas as pd
    with open(ahf_file) as f:
        header_line = f.readline()
    names = [n.split("(")[0] for n in header_line.lstrip("#").split()]
    df = pd.read_csv(ahf_file, sep=r"\s+", skiprows=1, names=names, header=None)

    row = df[df["ID"] == host_id]
    if len(row) != 1:
        raise ValueError(f"expected exactly 1 halo with ID={host_id}, got {len(row)}")

    xc, yc, zc = (row[["Xc", "Yc", "Zc"]].to_numpy()[0]) / hubble_h
    return xc, yc, zc


def get_scale_factor(ahf_file):
    """
    Extract the scale factor from the AHF filename's redshift, e.g.
    '...cdm.z0.474.AHF_halos' -> z=0.474 -> a=1/(1+z).
    Needed because star particle positions from the extraction pipeline
    are in PROPER kpc (atime hardcoded to 1, comoving->proper conversion
    disabled), while AHF's Xc/Yc/Zc are COMOVING kpc/h -- confirmed
    empirically: satellite offset from real subhalos grows with redshift
    and with distance from the projection center, and is corrected by
    multiplying (not dividing) the centered satellite positions by a.
    """
    import re
    m = re.search(r"z(\d+\.\d+)", ahf_file)
    if m is None:
        raise ValueError(f"could not parse redshift from {ahf_file}")
    z = float(m.group(1))
    return 1.0 / (1.0 + z)


def load_ahf_df(ahf_path):
    with open(ahf_path) as f:
        header_line = f.readline().lstrip("#").strip()
    names = [c.split("(")[0] for c in header_line.split()]
    return pd.read_csv(ahf_path, sep=r"\s+", skiprows=1, names=names)