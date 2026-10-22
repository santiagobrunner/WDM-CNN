#!/usr/bin/env python3
"""Mask the satellites in the mock images (what project_satellites.py was for).

For every halo, DM model, snapshot and line of sight: read the AHF satellites,
project them into the image, replace their pixels by local background noise,
and write the result next to the originals, in input_images_<sb>_f_masked/.

    python scripts/make_masked_images.py --images-root ROOT --base-path AHF_ROOT \\
        --lookup-csv resources/satellite_lookup.csv --sb 35 --halos FB15_h15_L11_r4_p1

Use --output-root with an empty folder to leave an existing masked dataset alone.
"""

import argparse
import os

import numpy as np
import pandas as pd
from astropy.io import fits

# names and signatures of the original project_satellites.py functions
from wdm_cnn.imaging.masking import (
    build_satellite_mask,
    mask_fill_with_noise_local_soft,
    satellite_mask_radii_pixels,
)
from wdm_cnn.imaging.projection import project_to_pixels
from wdm_cnn.simulation.ahf import (
    get_halo_center,
    get_scale_factor,
    read_satellites_by_radius,
    resolve_ahf_path,
    resolve_host_id,
)
from config import DEFAULT_SNAPSHOTS
from wdm_cnn.data.loaders import dm_models_for

dm_models = dm_models_for(5)
N_LOS = 24


def image_path(root, sb, halo, dm, snap, los, masked=False):
    halo_nr = int(halo.split("_")[1].replace("h", ""))
    folder = f"input_images_{sb}_f" + ("_masked" if masked else "")
    return f"{root}/{folder}/{halo}/{snap:03d}/SB_h{halo_nr}_{dm}_{snap:03d}_LOS{los:02d}_N{sb}.fits.gz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-root", required=True, help="folder containing the input_images_* folders")
    ap.add_argument("--output-root", help="where to write the masked folder (default: images-root)")
    ap.add_argument("--base-path", required=True, help="path preceding data/FIREbox/zooms/analysis/...")
    ap.add_argument("--lookup-csv", required=True, help="CSV with columns halo_nr,dm,snap,haloIndex,...")
    ap.add_argument("--sb", type=int, required=True, help="surface brightness limit of the images")
    ap.add_argument("--halos", nargs="+", required=True, help="FULL halo directory names")
    ap.add_argument("--dms", nargs="+", default=dm_models)
    ap.add_argument("--snaps", nargs="+", type=int, default=DEFAULT_SNAPSHOTS)
    ap.add_argument("--n-star-min", type=int, default=10, help="minimum stars for a satellite to be masked")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    lookup_df = pd.read_csv(args.lookup_csv)
    output_root = args.output_root or args.images_root

    for halo in args.halos:
        halo_nr = int(halo.split("_")[1].replace("h", ""))
        for dm in args.dms:
            for snap in args.snaps:
                if not os.path.exists(image_path(args.images_root, args.sb, halo, dm, snap, 0)):
                    continue  # this combination has no images

                # the satellites are the same for every line of sight: read the AHF file once
                ahf_path = resolve_ahf_path(args.base_path, halo, dm, snap)
                host_id = resolve_host_id(lookup_df, halo_nr, dm, snap)
                sat_pos, mstar = read_satellites_by_radius(ahf_path, host_id, n_star_min=args.n_star_min)
                a = get_scale_factor(ahf_path)
                sat_pos = (sat_pos - np.array(get_halo_center(ahf_path, host_id))) * a  # proper kpc
                z = 1.0 / a - 1.0

                for los in range(N_LOS):
                    in_path = image_path(args.images_root, args.sb, halo, dm, snap, los)
                    out_path = image_path(output_root, args.sb, halo, dm, snap, los, masked=True)
                    if os.path.exists(out_path) and not args.overwrite:
                        continue

                    image = fits.getdata(in_path)
                    hdr = fits.getheader(in_path)

                    masked, n_masked = image.copy(), 0
                    if len(sat_pos) > 0:
                        col, row = project_to_pixels(sat_pos, hdr)
                        radii = satellite_mask_radii_pixels(
                            mstar, z, hdr["OBJ_DIST"], hdr["PIXFOVX"], min_radius_pix=1.0)
                        mask = build_satellite_mask((hdr["NY"], hdr["NX"]), col, row, radii)
                        if mask.sum() > 0:
                            seed = [halo_nr, snap, los, args.dms.index(dm)]  # one fixed seed per image
                            masked, _ = mask_fill_with_noise_local_soft(
                                image, mask, col, row, radii, hdr["NOISESTD"], seed=seed)
                            masked = masked.astype(image.dtype)
                            n_masked = int(mask.sum())

                    hdr["MASKED"] = (True, "satellites replaced by local background noise")
                    hdr["NMASKPIX"] = (n_masked, "number of masked pixels")

                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    # write under a temporary name first, so an interrupted job never leaves a
                    # half-written file that the loader would silently skip
                    tmp = os.path.join(os.path.dirname(out_path), ".partial_" + os.path.basename(out_path))
                    fits.writeto(tmp, masked, header=hdr, overwrite=True)
                    os.replace(tmp, out_path)

                print(f"{halo} {dm} snap{snap:03d} done")


if __name__ == "__main__":
    main()