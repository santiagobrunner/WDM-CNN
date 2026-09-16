from pNbody.Mockimgs import luminosities, filters, instrument, ccd, telescope, obs_object
from astropy import units as u

mytelescope = telescope.Telescope(name='training_telescope', focal=153.6*u.mm)

myccd = ccd.CCD(name='training_ccd', shape=[256,256], pixel_size=[12.0*u.um, 12.0*u.um], fov=[1.146*u.deg, 1.146*u.deg])

myfilter = filters.Filter("BPASS230_Euclid_Y")

instrument = instrument.Instrument(name='training_instr_Y', 
                                  telescope=mytelescope, 
                                  ccd=myccd, 
                                  filter_type=myfilter)
