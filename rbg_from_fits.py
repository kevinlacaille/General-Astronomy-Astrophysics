import numpy as np
import aplpy as ap
#import montage_wrapper as montage
import pyfits as py
import pylab as pl
import img_scale
from astropy.visualization import make_lupton_rgb
import coords as co
#directory where the fits files are
dir = 'SPT0348_RGB_2/'



#the SPT0348 fits files: bands 3, 6, 7
#turning the nan's to numbers
#getting rid of the empy arrays (of size = 1)
b3 = np.nan_to_num(py.getdata(dir+'spt0348_b3ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_small_OG.fits')) #red: 3.6mm to 2.6mm
b6 = np.nan_to_num(py.getdata(dir+'spt0348_b6ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_2.fits')) #green: 1.4mm to 1.1mm
b7 = np.nan_to_num(py.getdata(dir+'spt0348_b7ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_2.fits')) #blue: 1.1mm to 0.8mm


#b3 is larger than the rest. resize b6,7 to be the size of b3
b6 = np.resize(b6_small,b3.shape)
b7 = np.resize(b7_small,b3.shape)

image = make_lupton_rgb(b3,b6,b7,Q=10,stretch=0.5)
pl.imshow(image, origin='lower')
pl.savefig('test_lupton.png')



############
# PYFITS WAY
############

#b3 is larger than the rest. resize b6,7 to be the size of b3
#b6 = np.resize(b6_small,b3.shape)
#b7 = np.resize(b7_small,b3.shape)

#setting up the image array
img = np.zeros((b3.shape[0], b3.shape[1], 3), dtype='uint8')

#set up the min and the max flux values
img[:,:,0] = img_scale.linear(b3, scale_min=0, scale_max=10000)
img[:,:,1] = img_scale.linear(b3, scale_min=0, scale_max=10000)
img[:,:,2] = img_scale.linear(b3, scale_min=0, scale_max=10000)

#show image

pl.clf()
pl.imshow(img, aspect='equal', origin='lower')
pl.title('SPT0348')
pl.savefig('rgb_pyfits.png')


#cubehelix

##############
# ASTROPY WAY
##############

#the SPT0348 fits files: bands 3, 6, 7
#b3 = dir+'spt0348_b3ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_2.fits' #red: 3.6mm to 2.6mm
#b6 = dir+'spt0348_b6ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_2.fits' #green: 1.4mm to 1.1mm
#b7 = dir+'spt0348_b7ctm_dirty_briggs_robust05_15klambda_uvtaperbeam_2.fits' #blue: 1.1mm to 0.8mm

b3 = 'best/spt0348_band3_clean1000_cont.fits'
b6 = 'best/spt0348_band6_clean1000_cont.fits'
b7 = 'best/spt0348_band7_clean1000_cont.fits'

#making a 3D cube from the bands

ap.make_rgb_cube([b3, b6, b7], 'spt0348_cube.fits')
ap.make_rgb_cube([b3, b6, b7], 'spt0348_cube')

##making an image from the cubes
ap.make_rgb_image(data='spt0348_cube.fits',output='spt0348_rgb_astropy.png', vmax_r=0.4*0.000553248, vmax_g=0.3*0.00875461, vmax_b=0.3*0.016697, vmin_r=1.2*-7.19859e-05,vmin_g=1.2*-0.000468956,vmin_b=1.2*-0.000726759)



###show the rgb image of SPT0348
#rgb = ap.FITSFigure('spt0348_cube_2d.fits')
##rgb.recenter(co.convDMS('3:48:42.312'), co.convHMS('-62:20:50.63'),width = 10.0/3600, height= 10.0/3600)
#rgb.show_colorscale(cmap='cubehelix', vmax=0.0005, vmin=-0.000537558, interpolation='nearest')
#rgb.save('test.png')
##rgb.show_rgb()
