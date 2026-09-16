#!/usr/bin/env python

"""Extract particles from a Gadget / GIZMO snapshot"""

import glob
import os
import errno
import sys
import argparse
import functools
import numpy as np
import h5py
import re

import auxiliary as aux

# ********************
# (c) R. Feldmann 2018
# Use at your own risk
# ********************

verbosity = 1
update_pickle = False

# xorshift64star random number generator
# from https://en.wikipedia.org/wiki/Xorshift
# https://github.com/jjbdx/xorshiftplus-c/blob/master/xorshift64star.c
class Xorshift64star(object):

    def __init__(self, state=1):
        # The state must be seeded with a nonzero value
        self.state = np.uint64(state)
        self.rand_max = np.uint64(2**64-1)

    def next(self):
        self.state ^= self.state >> np.uint64(12)
        self.state ^= self.state << np.uint64(25)
        self.state ^= self.state >> np.uint64(27)
        return np.multiply(self.state, np.uint64(2685821657736338717))

def int_to_int(x, N):
    r = Xorshift64star(state=x+1)
    for i in range(16):
        r.next()
    a = r.next()
    return np.uint64(a / (np.uint64(r.rand_max / N) + np.uint64(1)))

def shift_center(x, c, L=None):
    if c is not None:
        x = x - c
        if L is not None:
            sel = x > 0.5*L
            x[sel] -= L
            sel = x <= -0.5*L
            x[sel] += L
    return x

def in_sphere(x, c, r, L=None):
    xc = shift_center(x, c, L)
    return np.sum(np.square(xc), axis=1) < r*r

def in_box(x, c, r, L=None):
    xc = shift_center(x, c, L)
    return np.max(np.abs(xc), axis=1) < r

from astropy.cosmology import Planck15 as cosmo
aexps = np.linspace(1e-2, 1, 1000)
ages = cosmo.age(1./aexps-1.).value

def calc_age(a_curr, a_form):
    """return t_curr - t_form in yr"""
    t_curr = cosmo.age(1./a_curr - 1.).value
    t_form = np.interp(a_form, aexps, ages)
    return (t_curr - t_form)*1e9

def analyze_filename(filename, multiFile=False):
    filedir = os.path.dirname(filename)
    filebase = os.path.basename(filename)

    aux = os.path.splitext(filebase)
    fileext = aux[1]
    assert(fileext == '.hdf5')

    filestem = aux[0]

    if multiFile:
        aux = os.path.splitext(filestem)
        fileprefix = aux[0]
        fileindex = int(aux[1].lstrip('.'))
    else:
        fileprefix = filestem
        fileindex = None
	
    return fileprefix, fileindex, fileext, filedir

def create_header(fs, fd, name, node): 
    """copy header"""

    if isinstance(node, h5py.Group):
        if name == 'Header':
            fs.copy(name, fd, name=name)
            if verbosity > 1:
                print('{} copied'.format(name))
        else:
            if verbosity > 1:
                print('Group {} skipped'.format(name))

def copy_particles(fs, fd, length, gzip, mask, start_output, center,
                   rescale_radius, boxSize, samplingFactors, name, node):
    """copy all particles according to given mask"""

    compression = 'gzip'
    if gzip == -1:
        if hasattr(node, 'compression_opts'):
            compression_opts = node.compression_opts
        else:
            compression_opts = None
    elif gzip == 0:
        compression_opts = None
    else:
        compression_opts = gzip

    if compression_opts is None:
        compression = None

    if isinstance(node, h5py.Dataset):
        ptype = get_ptype(name)

        samplingFactor = samplingFactors[ptype]

        if ptype not in mask:
            return 

        ncopy = np.sum(mask[ptype]).astype('uint64')

        if ncopy == 0:
            return

        # create dataset if needed
        if ptype not in length:
            return

        if length[ptype] == 0:
            return

        if name not in fd:

            shape_dst = np.copy(node.shape)

            shape_dst[0] = length[ptype]

            if compression is None:
                dnew = fd.create_dataset(
                    name, shape=shape_dst, dtype=node.dtype, chunks=None)
            else:
                dnew = fd.create_dataset(
                    name, shape=shape_dst, dtype=node.dtype, chunks=True, 
                    shuffle=True, compression=compression,
                    compression_opts=compression_opts, fletcher32=True)

            if verbosity > 1:
                print('Dataset {} created with shape {}'.format(name, shape_dst))

        start = start_output[ptype]

        # for performance reasons read slabs into memory 
        slab_length = 1000000
        slab_n = int(np.ceil(node.shape[0] / slab_length))
        l_start = start
        for slab_i in range(slab_n):
            slab_start = slab_i * slab_length
            slab_end = min((slab_i+1) * slab_length, node.shape[0])
            slab_node = np.array(node[slab_start:slab_end])

            l_ncopy = np.sum(mask[ptype][slab_start:slab_end]).astype(
                'uint64')
            if l_ncopy == 0:
                continue

            l_end = l_start + l_ncopy
            if node.ndim == 1:
                if 'Coordinates' in name and center is not None:
                    aux = shift_center(
                        slab_node[mask[ptype][slab_start:slab_end]],
                        center, boxSize) / rescale_radius
                elif 'Masses' in name and samplingFactor > 1:
                    aux = samplingFactor * slab_node[
                        mask[ptype][slab_start:slab_end]]
                else:
                    aux = slab_node[mask[ptype][slab_start:slab_end]]
                fd[name][l_start:l_end] = aux 
            else:
                if 'Coordinates' in name and center is not None:
                    aux = shift_center(
                        slab_node[mask[ptype][slab_start:slab_end], :],
                        center, boxSize) / rescale_radius
                else:
                    aux = slab_node[mask[ptype][slab_start:slab_end], :]
                fd[name][l_start:l_end, :] = aux 

            l_start = l_end 

        if verbosity > 1:
            print('Dataset {}: copied {} of {} entries'.format(
                name, ncopy, len(mask[ptype])))


def get_ptype(s):
    m = re.search(r'PartType([0-9]+)', s)
    return int(m.group(1))


def read_header(path, index):
    fname = get_filename(path, index) 
    with h5py.File(fname, 'r') as file:
        aux = file["Header"].attrs
        header = dict(zip(aux.keys(), aux.values()))
    return header


def get_filenames(path, sim, snap_id, suffix=''):
    filenames = glob.glob('{}/{}/snapdir_{:03d}/snapshot_{:03d}{}.*.hdf5'.format(
        path, sim, snap_id, snap_id, suffix))
    if len(filenames) == 0:
        filenames = glob.glob('{}/{}/{:03d}/snapshot_{:03d}{}.*.hdf5'.format(
            path, sim, snap_id, snap_id, suffix))
    if len(filenames) == 0:
        filenames = glob.glob('{}/{}/snapshot_{:03d}{}.*.hdf5'.format(
            path, sim, snap_id, suffix))
    if len(filenames) == 0:
        filenames = glob.glob('{}/{}/{:03d}/snapshot_{:03d}{}.hdf5'.format(
            path, sim, snap_id, snap_id, suffix))
    if len(filenames) == 0:
        filenames = glob.glob('{}/{}/snapshot_{:03d}{}.hdf5'.format(
            path, sim, snap_id, suffix))
    if len(filenames) == 0:
        raise Exception('No snapshot file found -- Abort')

    # sort filenames numerically
    def num_key(s):
        m = re.search(r'\.([0-9]+)\.hdf5$', s)
        return int(m.group(1))
    if len(filenames) > 1:
        filenames.sort(key=num_key)

    if verbosity > 1:
        print('input filenames = {}'.format(filenames), flush=True)
    for fname in filenames:
        if not os.path.isfile(fname):
            raise Exception('Input file {} not found'.format(fname))

    return filenames


def main(sim, snap_id, path, AHF_path, VA_path, ofilename='out',
         components_str='111111', samplingFactors=np.ones(6, dtype=int),
         region='sphere', periodic=True, haloID=-1, haloIndex=-1, split=False,
         copyH2=False, t_age=0, center=None, radius=1., minradius=0.,
         recenter=False, rescale=False, gzip=-1):

    components = int(components_str, 2)
    ptypes = []
    for i in range(6):
        if components & (1 << (5-i)): # check if i'th bit is set
            ptypes.append(i)

    if haloID > -1 or haloIndex > -1:
        halos, nhalos, haloID_to_haloIndex = aux.get_halos(
            sim, snap_id, AHF_path, VA_path, verbosity, use_pickle=True,
            update_pickle=update_pickle)
        if np.max(halos['ID']) > 2**32:
            halo_central = halos['hostHalo'] == 0
        else:
            halo_central = halos['hostHalo'] == -1

        if 'Mhalo' in halos.keys():
            halos.rename(columns={'Mhalo': 'Mvir'}, inplace=True, errors='raise')
        if 'Rhalo' in halos.keys():
            halos.rename(columns={'Rhalo': 'Rvir'}, inplace=True, errors='raise')

        if haloIndex < 0:
            haloIndex = haloID_to_haloIndex[haloID]
        if haloID < 0:
            haloID = halos['ID'][haloIndex]

        center = np.array((halos['Xc'][haloIndex], halos['Yc'][haloIndex],
                          halos['Zc'][haloIndex]))
        if radius > 0:
            radius = radius * halos['Rvir'][haloIndex]
        elif radius == 0:
            if verbosity:
                print('Reading sizes', flush=True)
            size_filename = '{}/{}/{:03d}/Rstar.hdf5'.format(
                VA_path, sim, snap_id)
            with h5py.File(size_filename, 'r') as f:
                R_aux = f['Rstar_tot_1'][()]
                for ihaloID, l_haloID in enumerate(f['haloIDs'][()]):
                    if l_haloID == haloID:
                        radius = R_aux[ihaloID]
        else:
            radius = -radius
        radius = max(radius, minradius)

        is_central_label = 'main '
        if not halo_central[haloIndex]:
            is_central_label = 'sub-'
        print('Extracting {}halo # {} with ID {} at position {:5g} {:5g} {:5g} '.format(
            is_central_label, haloIndex, haloID, center[0], center[1], center[2]), end='')
        print('and radius {}'.format(radius))

    if split:
        ofilenames = ['{}_wosub.hdf5'.format(ofilename),
                      '{}_wsub.hdf5'.format(ofilename),
                      '{}_sub.hdf5'.format(ofilename)]
    else:
        ofilenames = ['{}.hdf5'.format(ofilename)]

    filenames = get_filenames(path, sim, snap_id)

    if split:
        VA_filenames = get_filenames(VA_path, sim, snap_id, suffix='_VA')
        assert len(filenames) == len(VA_filenames)

    if copyH2:
        try:
            H2_filenames = get_filenames(VA_path, sim, snap_id, suffix='_H2')
            assert len(filenames) == len(H2_filenames)
        except Exception:
            print('Warning: No matching H2 files found')
            copyH2 = False
    if not copyH2:
        print('H2 data (if present) will not be copied')

    for iofile, ofilename in enumerate(ofilenames):

        # create output file
        if os.path.isfile(ofilename):
            print('Warning: Overwriting existing output file {}'.format(
                ofilename))

        if verbosity > 0:
            print('Creating output file {}'.format(
                ofilename), flush=True)
        f_dst = h5py.File(ofilename, 'w') 

        selected = np.zeros((len(filenames), 6), dtype=np.uint64)
        numparts_ifile = np.zeros((len(filenames), 6), dtype=np.uint64)

        # calculate how many particles selected and do some checks
        for ifile, filename in enumerate(filenames):
            fs = h5py.File(filename, 'r')
            numparts_ifile[ifile, :] = fs['Header'].attrs['NumPart_ThisFile']
            if verbosity > 0:
                print('reading file {}'.format(filename), flush=True)

            if split:
                fsVA = h5py.File(VA_filenames[ifile], 'r')

            if copyH2:
                fsH2 = h5py.File(H2_filenames[ifile], 'r')

            # some checks 
            numfiles = fs['Header'].attrs['NumFilesPerSnapshot']
            assert numfiles == len(filenames)

            if ifile == 0:
                numparts_lowword = fs['Header'].attrs['NumPart_Total']
                numparts_highword = fs['Header'].attrs['NumPart_Total_HighWord']
                boxSize = None
                if periodic:
                    try:
                        boxSize = fs['Header'].attrs['BoxSize']
                    except:
                        pass
            else:
                assert np.all(
                    numparts_lowword == fs['Header'].attrs['NumPart_Total'])
                assert np.all(
                    numparts_highword == fs[
                        'Header'].attrs['NumPart_Total_HighWord'])

            for ptype in ptypes:
                if 'PartType{}'.format(ptype) not in fs:
                    continue

                pos = fs['/PartType{}/Coordinates'.format(ptype)]

                if pos.shape[0] == 0:  # no particles
                    continue

                samplingFactor = samplingFactors[ptype]
                
                if samplingFactor > 1:
                    ids = fs['/PartType{}/ParticleIDs'.format(ptype)]
                    sampling_sel = (int_to_int(np.array(ids), samplingFactor) == 0)
                else:
                    sampling_sel = np.ones(pos.shape[0], dtype=bool)

                if split:
                    haloCounts = fsVA[
                        '/PartType{}/HaloCounts'.format(ptype)]
                    haloIDs_all = fsVA[
                        '/PartType{}/HaloIDs'.format(ptype)]
                    if iofile == 1 or iofile == 2:
                        haloIDs_host = fsVA[
                            '/PartType{}/HaloIDs_host'.format(ptype)]

                    if iofile == 0:   # current w/o sats (but may incl super)
                        sel_split = ((haloCounts[:] > 0) &
                                     (haloIDs_all[:] == haloID))
                    elif iofile == 1: # current w/ sats (& w/ super) 
                        sel_split = ((haloCounts[:] > 0) &
                                     ((haloIDs_host[:] == haloID) |
                                      (haloIDs_all[:] == haloID)))
                    elif iofile == 2: # sats of current central halo
                        sel_split = ((haloCounts[:] > 0) &
                                     (haloIDs_all[:] != haloID) &
                                     (haloIDs_host[:] == haloID))

                    sampling_sel &= sel_split

                if ptype == 4 and t_age > 0:
                    a_form = fs[
                        '/PartType{}/StellarFormationTime'.format(ptype)][:]
                    age = calc_age(fs['Header'].attrs['Time'], a_form)
                    sampling_sel &= (age <= t_age)

                if region == 'sphere':
                    selected[ifile, ptype] = np.sum(
                        in_sphere(pos, center, radius, boxSize) & sampling_sel)
                elif region == 'box':
                    selected[ifile, ptype] = np.sum(
                        in_box(pos, center, radius, boxSize) & sampling_sel)

            fs.close()
            if split:
                fsVA.close()
            if copyH2:
                fsH2.close()

            filebase, fileindex, fileext, filedir = analyze_filename(
                filename, numfiles > 1)

        # extract particles
        numparts_ofile = np.sum(selected, axis=0)
        if verbosity > 0:
            print('Extract {} particles'.format(numparts_ofile))

        numparts_ofile_dict = {}
        for ptype in ptypes:
            numparts_ofile_dict[ptype] = numparts_ofile[ptype]
        start_particles = np.zeros((len(filenames), 6), dtype=np.uint64)
        start_particles[1:, :] = np.cumsum(selected, axis=0)[:-1, :]

        for ifile, filename in enumerate(filenames):

            f_src = h5py.File(filename, 'r')
            if split:
                fsVA = h5py.File(VA_filenames[ifile], 'r')
            if copyH2:
                fsH2 = h5py.File(H2_filenames[ifile], 'r')

            boxSize = None
            if periodic:
                try:
                    boxSize = f_src['Header'].attrs['BoxSize']
                except:
                    pass 

            if ifile == 0:
                create_datasets = functools.partial(
                    create_header, f_src, f_dst)
                f_src.visititems(create_datasets)

                if verbosity > 0:
                    print('Adding fields to output file {}'.format(
                        os.path.basename(ofilename)), flush=True)

            if np.sum(selected[ifile]) == 0:
                continue

            if verbosity > 0:
                print('From input file {}, copy {} particles'.format(
                    os.path.basename(filename), selected[ifile, :]), flush=True)

            mask = {}
            for ptype in ptypes:
                if 'PartType{}'.format(ptype) not in f_src:
                    continue

                pos = f_src['/PartType{}/Coordinates'.format(ptype)]

                if pos.shape[0] == 0:  # no particles
                    continue

                samplingFactor = samplingFactors[ptype]
                if samplingFactor > 1:
                    ids = f_src['/PartType{}/ParticleIDs'.format(ptype)]
                    sampling_sel = (int_to_int(np.array(ids), samplingFactor) == 0)
                else:
                    sampling_sel = np.ones(pos.shape[0], dtype=bool)

                if split:
                    haloCounts = fsVA[
                        '/PartType{}/HaloCounts'.format(ptype)]
                    haloIDs_all = fsVA[
                        '/PartType{}/HaloIDs'.format(ptype)]
                    if iofile == 1 or iofile == 2:
                        haloIDs_host = fsVA[
                            '/PartType{}/HaloIDs_host'.format(ptype)]

                    if iofile == 0:   # current w/o sats (but may incl super)
                        sel_split = ((haloCounts[:] > 0) &
                                     (haloIDs_all[:] == haloID))
                    elif iofile == 1: # current w/ sats (& w/ super) 
                        sel_split = ((haloCounts[:] > 0) &
                                     ((haloIDs_host[:] == haloID) |
                                      (haloIDs_all[:] == haloID)))
                    elif iofile == 2: # sats of current central halo
                        sel_split = ((haloCounts[:] > 0) &
                                     (haloIDs_all[:] != haloID) &
                                     (haloIDs_host[:] == haloID))

                    sampling_sel &= sel_split

                if ptype == 4 and t_age > 0:
                    a_form = f_src[
                        '/PartType{}/StellarFormationTime'.format(ptype)]
                    age = calc_age(f_src['Header'].attrs['Time'], a_form[:])
                    sampling_sel &= (age <= t_age)

                if region == 'sphere':
                    mask[ptype] = (in_sphere(pos, center, radius, boxSize) &
                        sampling_sel)
                elif region == 'box':
                    mask[ptype] = (in_box(pos, center, radius, boxSize) &
                        sampling_sel)

            if rescale:
                rescale_radius = radius
            else:
                rescale_radius = 1
            if recenter:
                recenter_center = center
            else:
                recenter_center = None

            update_datasets = functools.partial(
                copy_particles, f_src, f_dst, numparts_ofile_dict, gzip, mask,
                start_particles[ifile], recenter_center, rescale_radius,
                boxSize, samplingFactors)

            f_src.visititems(update_datasets)
            f_src.close()

            if copyH2:
                update_datasets = functools.partial(
                    copy_particles, fsH2, f_dst, numparts_ofile_dict, gzip, mask,
                    start_particles[ifile], recenter_center, rescale_radius,
                    boxSize, samplingFactors)
                fsH2.visititems(update_datasets)
                fsH2.close()

            if split:
                fsVA.close()


        # update header
        f_dst['Header'].attrs[
            'NumPart_ThisFile'] = numparts_ofile.astype('int32')
        if verbosity > 1:
            print('Attribute Header.NumPart_ThisFile adjusted')

        f_dst['Header'].attrs[
            'NumPart_Total'] = numparts_ofile.astype('int32')
        f_dst['Header'].attrs[
            'NumPart_Total_HighWord'] = (numparts_ofile >> 32).astype('int32')
        if verbosity > 1:
            print('Attribute Header.NumPart_Total adjusted')
            print('Attribute Header.NumPart_Total_HighWord adjusted', flush=True)

        if numfiles != 1:
            f_dst['Header'].attrs['NumFilesPerSnapshot'] = np.int32(1)
            if verbosity > 1:
                print('Attribute Header.NumFilesPerSnapshot adjusted', flush=True)
        f_dst.close()

        if verbosity > 0:
            print('==> In total, {} particles copied to output file {}'.format(
                numparts_ofile, os.path.basename(ofilename)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract particles within a spherical or cubical region ' 
                    'from a Gadget/GIZMO HDF5 snapshot (c) R. Feldmann 2018')
    parser.add_argument('simulation', help='simulation name ')
    parser.add_argument('-n', '--index', help='snapshot index, default 0',
                        default=0)
    parser.add_argument('-P', '--path',
                        help='path to the simulation, default '
                        '"/net/cephfs/shares/feldmann.ics.mnf.uzh/data/FIREbox'
                        '/production_runs"', default='/net/cephfs/shares/'
                        'feldmann.ics.mnf.uzh/data/FIREbox/production_runs')
    parser.add_argument('-A', '--AHFPath', help='path for AHF files, '
                        'default="/net/cephfs/shares/feldmann.ics.mnf.uzh/'
                        'data/FIREbox/analysis/AHF"', default='/net/cephfs/'
                        'shares/feldmann.ics.mnf.uzh/data/FIREbox/analysis/'
                        'AHF')
    parser.add_argument('-D', '--valueAddedPath', help='path for value-added '
                        'files, default="/net/cephfs/shares/'
                        'feldmann.ics.mnf.uzh/data/FIREbox/analysis/'
                        'value_added"', default='/net/cephfs/shares/'
                        'feldmann.ics.mnf.uzh/data/FIREbox/analysis/'
                        'value_added')
    parser.add_argument('-o', '--outfilename',
                        help='prefix of output file, '
                        'default "out"', default='out')
    parser.add_argument('-R', '--region',
                        help='"sphere" or "box", default "sphere"',
                        default='sphere')
    parser.add_argument('-c', '--components',
                        help='which components are to be selected (Gadget '
                        'notation), default "111111" (all)', default='111111')
    parser.add_argument('-S', '--samplingFactor',
                        help='downsampling by this integer, default 1 (off); '
                        'masses are increased by this factor, no other '
                        'particle property is explicitly changed; the '
                        'samplingFactor can be a list of integers e.g., '
                        '"64 64 64 64 8 1" to downsample different components '
                        'at a different amount', default='1')
    parser.add_argument('-C', '--recenter', action='store_true',
                        help=('if set change coordinates in output s.t. '
                              'center is at (0, 0, 0), default not set'),
                        default=False)
    parser.add_argument('-L', '--rescale', action='store_true',
                        help=('change coordinates s.t. +-radius corresponds '
                              'to +-1, if set implies --recenter, '
                              'default not set'),
                        default=False)
    parser.add_argument('--periodic', 
                        help=('coordinates are periodic with box-size stored '
                              'in snapshot file, default True'),
                        default=True)
    parser.add_argument('-i', '--ID', help=('halo ID according to AHF halo '
                        'file; if ID is given, set x/y/z and r, '
                        'default -1 (do not use)'), default=-1)
    parser.add_argument('--haloIndex', help=('halo index according to AHF '
                        'halo file; if index is given overwrite value of ID '
                        'default -1 (do not use)'), default=-1)
    parser.add_argument('-s', '--split', help=('split output into main and '
                        'sub-structures, default=False'), default=False)
    parser.add_argument('--copyH2', help=('store fH2, default=False'),
                        default=False)
    parser.add_argument('-a', '--age', help=('maximum age in yr of star '
                        'particles, default=0 (off)'), default=0)
    parser.add_argument('-x', '--xcenter',
                        help='provide x coordinate of center, default 0.',
                        default=0.)
    parser.add_argument('-y', '--ycenter',
                        help='provide y coordinate of center, default 0.',
                        default=0.)
    parser.add_argument('-z', '--zcenter',
                        help='provide z coordinate of center, default 0.',
                        default=0.)
    parser.add_argument('-r', '--radius',
                        help='provide radius of sphere / half box-length; if '
                        '--ID >=0: if positive it is radius in units of Rvir, '
                        'if negative it is -radius in ckpc/h, if 0 use '
                        'stellar half-mass radius; default 1.', default=1.)
    parser.add_argument('--minradius',
                        help='minimum radius of sphere / half box-length, '
                        'i.e., actual radius is max(radius, minradius), '
                        'in ckpc/h; default 0.', default=0.)
    parser.add_argument('-g', '--gzip',
                        help='gzip compression level: -1 (set by source '
                        'file), 0 (turn off), 1-9 (gzip levels), '
                        'default -1', default=-1)
    parser.add_argument('-V', '--verbose',
                        help='verbosity: 0(silent)-2, default 1',
                        default=1)

    args = parser.parse_args()
    verbosity = int(args.verbose)

    samplingFactors = list(map(int, args.samplingFactor.split()))
    if len(samplingFactors) != 1 and len(samplingFactors) != 6:
        raise('parameter samplingFactor needs to be a single integer or a '
              'list of length 6')
    if len(samplingFactors) == 1:
        samplingFactors = samplingFactors[0]*np.ones(6, dtype=int)
    else:
        samplingFactors = np.array(samplingFactors)

    recenter = int(args.recenter)
    rescale =  int(args.rescale)
    if rescale:
        recenter = True

    center = np.array(
        (float(args.xcenter), float(args.ycenter), float(args.zcenter)))

    main(args.simulation, int(args.index), args.path, args.AHFPath,
         args.valueAddedPath, args.outfilename, args.components,
         samplingFactors, args.region, int(args.periodic), int(args.ID),
         int(args.haloIndex), int(args.split), int(args.copyH2),
         float(args.age), center, float(args.radius), float(args.minradius),
         recenter, rescale, gzip=int(args.gzip))
