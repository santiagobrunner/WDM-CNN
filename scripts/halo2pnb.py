import os
import numpy as np
import pandas as pd
import glob
import h5py

from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.io import ascii
from astropy import units as u
from astropy.wcs import WCS

from pNbody import *

# (c) L. Tortora, R. Feldmann

# from rf_helper.py
def get_angular_momentum(x, y, z, vx, vy, vz, m, radius=3, normalize=False):
    """Return angular momentum vector

    Keyword arguments:
    x,y,z,vx,vy,vz,m -- lists of position, velocity, mass of particles
    radius           -- only include particles in this radius around [0,0,0]
    normalize        -- if True, normalize length of ang. mom. vector to 1
    """
    # copy references
    for q in [x, y, z, vx, vy, vz]:
        q = np.array(q, dtype='d')+1.0e-10
    # copy list because modify below
    m = np.array(m, dtype='d', copy=True)+1.0e-10

    r = np.sqrt(x*x + y*y + z*z)
    nstars = sum(r <= radius)
    if nstars == 0:
        return [0, 0, 1], 0

    m[r > radius] = 0
    pos = np.zeros([x.size, 3], dtype='d')
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = z
    vel = np.zeros([vx.size, 3], dtype='d')
    vel[:, 0] = vx
    vel[:, 1] = vy
    vel[:, 2] = vz

    j = np.cross(pos, vel)
    j[:, 0] *= m
    j[:, 1] *= m
    j[:, 2] *= m
    j_tot = np.sum(j, axis=0)

    if normalize:
        j_tot /= np.linalg.norm(j_tot)

    return j_tot, nstars

def compute_angular_momentum(x, y, z, vx, vy, vz, m, c):
    """Compute ang. mom. of young stars within some small radius"""
    age_max = 0.
    while True:
        age_max += 2e8
        ix = c*1e9 < age_max  # ages less than 200 Myr
        L, nstars = get_angular_momentum(
            x[ix], y[ix], z[ix], vx[ix], vy[ix],
            vz[ix], m[ix], radius=2.5, normalize=True)
        if nstars >= 100:
            break
        if age_max > 3e9:
            if nstars < 10:
                L = [0, 0, 1]
            break
    return L

def compute_view_core(viewmode='random', L=np.array([0, 1., 0])):
    """Compute view & up vector for a provided viewmode, e.g, 'faceon'"""

    if viewmode == 'random':
        view = get_random_vectors(1)
        up = np.array([0, 0, 1])

    elif viewmode == 'faceon' or viewmode == 'edgeon':
        if viewmode == 'faceon':
            view = L
            up = np.array([0, 0, 1])
        elif viewmode == 'edgeon':
            raise Exception('viewmode == edgeon not available')

    elif viewmode == 'z':
        view = np.array([0., 0., 1.])
        up = np.array([1., 0., 0.])

    elif viewmode == 'x':
        view = np.array([1., 0., 0.])
        up = np.array([0., 1., 0.])

    elif viewmode == 'y':
        view = np.array([0., 1., 0.])
        up = np.array([0., 0., 1.])

    else:
        raise ValueError('Viewmode %s is not supported by '
                         'compute_view(...)' % (viewmode))

    return view, up

def coordinates_rotate2(x_all, y_all, z_all, view=np.array([0., 0., 1.]),
                        up=np.array([1., 0., 0.])):
    """Rotate coordinates according to provided camera view and up vectors

    Keyword arguments:
    x_all, y_all, z_all -- numpy arrays of x, y, z positions of particles
    view                -- numpy 3 vector of a normalized vector that
                           points TOWARD the camera
    up                  -- numpy 3 vector of a normalized vector that
                           provides an 'up' direction
    the default choice view = [0, 0, 1], up = [1, 0, 0] rotates nothing
    Return numpy arrays of the new x, y, z positions of particles

    up & view follows approximately the tipsy convention
    tipsy-xall: view = [1, 0, 0], up = [0, 0, 1]
    tipsy-yall: view = [0, 1, 0], up = [0, 0, 1]
    tipsy-zall: view = [0, 0, 1], up = [0, 1, 0]

    Internally, create a right hand system up-right-view, where
    right is a unit vector proportional to view x up. Then map from
    x-y-z to up-right-view.
    """

    # normalize vectors (just to be sure)
    view = view/np.linalg.norm(view)
    up = up/np.linalg.norm(up)

    # create right-handed coordinate basis: up, right, view
    right = np.cross(view, up)
    if np.linalg.norm(right) < 1e-10:  # degenerate => manually build up vector
        if view[1] or view[2]:
            up = np.array([0, -view[2], view[1]])
        else:
            up = np.array([-view[1], view[0], 0])

        up = up/np.linalg.norm(up)
        right = np.cross(view, up)
        right = right/np.linalg.norm(right)
    else:                              # make sure up is orthogonal to view
        right = right/np.linalg.norm(right)
        up = np.cross(right, view)

    # rotate from up-right-view to x-y-z:
    # the rotation matrix is simply a matrix with up-right-view as columns
    R = np.array([up, right, view])

    x_new = R[0, 0] * x_all
    y_new = R[1, 0] * x_all
    z_new = R[2, 0] * x_all

    x_new += R[0, 1] * y_all
    y_new += R[1, 1] * y_all
    z_new += R[2, 1] * y_all

    x_new += R[0, 2] * z_all
    y_new += R[1, 2] * z_all
    z_new += R[2, 2] * z_all

    return x_new, y_new, z_new

def ensure_periodic(x, box_size):
    sel = x <= -box_size / 2
    x[sel] += box_size
    sel = x > box_size / 2
    x[sel] -= box_size
    return x

def main(input_file, output_file, h, Minit, faceon):

    # define units
    u_Length   = 1* u.kpc                 # Lengths in kpc
    u_Mass     = 10**10 * u.M_sun         # Masses in 1e10 solar masses
    u_Velocity = 1* u.km/u.s              # Velocities in km / s
    u_Time     = 1* u.Gyr                 # in Gyr
    toMsol     = u_Mass.to(u.M_sun).value

    with h5py.File(input_file, 'r') as f:
        a = f['Header'].attrs['Time'] # NEW
        
        x  = np.array([row for row in f['PartType4']['Coordinates'][:,0]]) / h # in ckpc
        y  = np.array([row for row in f['PartType4']['Coordinates'][:,1]]) / h # in ckpc
        z  = np.array([row for row in f['PartType4']['Coordinates'][:,2]]) / h # in ckpc

        vx = np.array([row for row in f['PartType4']['Velocities'][:,0]]) # in km / s
        vy = np.array([row for row in f['PartType4']['Velocities'][:,1]]) # in km / s
        vz = np.array([row for row in f['PartType4']['Velocities'][:,2]]) # in km / s

        pID = np.array(f['PartType4']['ParticleIDs'])
        cID = np.array(f['PartType4']['ParticleChildIDsNumber'])
        gID = np.array(f['PartType4']['ParticleIDGenerationNumber'])

        mass = np.ones(x.size) * Minit / 1e10 # initial mass of stellar particles in 1e10 solar masses

        if f['PartType4']['Metallicity'].ndim == 1:
            mz = np.log10(np.array(f['PartType4']['Metallicity']) / 0.02)

        elif f['PartType4']['Metallicity'].ndim == 2:
            mz = np.log10(np.array([row for row in f['PartType4']['Metallicity'][:,0]]) / 0.02) # metallicity in [M/H]
        
        # Age is hard, let's try; stellarformationtime has the scale factor at creation of the stellar particle
        zstars = 1 / np.array(f['PartType4']['StellarFormationTime']) - 1 # convert scale factor to redshift: 1+z = 1/a

        # Set-up FIREbox cosmology
        cosmo1 = FlatLambdaCDM(H0=67.74, Om0=0.3089, Ob0=0.0486) # Is this comsology correct?

        # Age of stellar particles is the lookback time from their formation
        age = np.array(cosmo1.lookback_time(zstars)) # age in Gyr

        if faceon:
            print('Performing rotation')
            # Perform rotation
            l_max = 50
            valid_indices = x**2 + y**2 + z**2 < l_max**2

            # calc angular momentum of <200 Myr stars within 2.5 kpc
            L = compute_angular_momentum(
                x[valid_indices], y[valid_indices], z[valid_indices],
                vx[valid_indices], vy[valid_indices], vz[valid_indices],
                mass[valid_indices], age[valid_indices])
            print(L)

            #FACEON
            view1, up1 = compute_view_core('faceon', L)
            x_new, y_new, z_new = coordinates_rotate2(x, y, z, view1, up1)
            vx_new, vy_new, vz_new = coordinates_rotate2(vx, vy, vz, view1, up1)

            coords = np.array([x_new, y_new, z_new])
            vels = np.array([vx_new, vy_new, vz_new])
        else:
            coords = np.array([x, y, z])
            vels = np.array([vx, vy, vz])

        coords = np.transpose(coords)
        vels = np.transpose(vels)

    # Create the pNbody object
    nb = Nbody(status='new',pos=coords,vel=vels,mass=mass,ftype='swift')
    nb.set_tpe(4) # set all particles to stellar particles (PartType4 for some reason)

    nb.age = age # units should already be correct
    nb.mh  = mz  # units should already be correct

    nb.num = pID
    nb.childnum = cID
    nb.gennum = gID

    ## This is all copied from Yves's script
    # add units
    nb.UnitLength_in_cm         = u_Length.to(u.cm).value
    nb.UnitMass_in_g            = u_Mass.to(u.g).value
    nb.UnitVelocity_in_cm_per_s = u_Velocity.to(u.cm/u.s).value
    nb.Unit_time_in_cgs         = u_Time.to(u.s).value

    nb.hubblefactorcorrection      = False  #Correction already applied in this file 
    nb.comovingtoproperconversion  = True#False
    #nb.atime                       = 1
    nb.atime                        = a
    nb.scalefactor                  = a
    nb.time                         = a
    nb.cosmorun                     = 1

    nb.ComputeRsp(Nngb=5)

    nb.rename(output_file)
    nb.write()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
    description=('Extract star particles from GIZMO file and store them '
                 'a format expected by pNbody (c) RF & Lucas Tortora 2024'))
    parser.add_argument('infilename', help='HDF5 filename')
    parser.add_argument('-f', '--faceon',
                        help='rotate coordinate system, default 0(no)',
                        default=0)
    parser.add_argument('-o', '--outprefix',
                        help='prefix of output file, default "pnbody"',
                        default='pnbody')
    args = parser.parse_args()
        
    """
    The input file should be the extracted halo in hdf5 format
    The output file is the 'Nbody' object used to make SB
    """
    base_dir = '.'
    input_dir = f'{base_dir}'
    output_dir = f'{base_dir}/to_pNbody'

    infilename = args.infilename
    faceon = int(args.faceon)
    outprefix = args.outprefix

    input_file  = f'{input_dir}/{infilename}'
    output_file = f'{output_dir}/{outprefix}_{infilename}'

    h = 0.6774
    boxsize = 15000 / h  # [ckpc]
    Minit = 6.26e4 / 8   # initial mass of star particles [Msun]

    # Get coordinates, rotate, delete original hdf5 (memory saving)
    main(input_file, output_file, h, Minit, faceon)
