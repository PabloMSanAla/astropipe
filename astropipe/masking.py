''' 
Module to create mask for a individual object (typicially a galaxy)
in astronomical images. It uses some of the following software:
   - SExtractor
   - NoiseChisel [Gnuastro]
   - MTObjects
'''
import os
import cv2
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.wcs import WCS, utils
import astropy.units as u
from astropy.coordinates import Angle

from scipy.ndimage import gaussian_filter
from photutils.segmentation import detect_sources
from photutils.aperture import CircularAperture
from fabada import fabada


from .classes import SExtractor, AstroGNU, MTObjects
from .utils import where, change_coordinates

from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter

from matplotlib import pyplot as plt 
from scipy.signal import convolve2d
from astropy.io import fits


point_sexcofing = {
                "CHECKIMAGE_TYPE": "SEGMENTATION",
                "DEBLEND_MINCONT": 0.005,
                "DEBLEND_NTHRESH": 32,
                "BACK_SIZE": 120,       # 20 
                'DETECT_THRESH':0.9,
                "SATUR_LEVEL": 50000.0,
                "MAG_ZEROPOINT": 22.5,
                'PHOT_FLUXFRAC': 0.5,
                'MEMORY_OBJSTACK': 10000,           # number of objects in stack
                'MEMORY_PIXSTACK': 1000000,         # number of pixels in stack
                'MEMORY_BUFSIZE': 51200
            }

def sigma_filter(catalog, columns, sigma=5, weights=None, colid = 'NUMBER'):
    '''
    Given a catalog and a colum ID, it filters values that are at a certain
    distance of the standard deviation compute using sigma clipped statistics.

    Parameters
    ----------
        catalog : astropy.table.Table
            Catalog with the data
        columns : list
            List of columns to filter
        sigma : float, optional
            Number of sigma to filter
        weights : array, optional
            Array of weights to apply to the columns values
        colid : str, optional
            Column ID of the catalog
    '''
    if weights is None: weights = np.ones(len(catalog))
    index = np.ones(len(catalog)).astype(bool)
    for col in columns:
        mean, median, std = sigma_clipped_stats(catalog[col], sigma=2.5)
        index &= catalog[col]*weights > mean + sigma*std
    return [catalog[colid][ind] for ind in np.where(index)[0]]

def gaussian1D(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian2D(l, sig):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def gaussian2d(positions, center, sigma):
    return np.exp(-((positions[0] - center[0])**2. / (2. * sigma[0]**2.) + (positions[1] - center[1] )**2. / (2. * sigma[1]**2.)))


def filter_distance2D(positions, center, pa, eps, distance):
    newCoordinates = change_coordinates(positions, center, pa)
    index = (newCoordinates[0] < distance) + (
             newCoordinates[1] < (1-eps)*distance)
    return index

def weight_distance2D(positions, center, pa, eps, sigma):
    newCoordinates = change_coordinates(positions, center, pa)
    return gaussian2d(newCoordinates, np.array([0,0]) , np.array([sigma,(1-eps)*sigma]))


def get_peaks(data, mask=None, verbose=False):
    original_stats = sigma_clipped_stats(data, mask=mask, sigma=2.5)
    recover = fabada(data, 2*(original_stats[2]**2),verbose=verbose)
    peaks = recover - gaussian_filter(data,sigma=5)
    mean, median, std = sigma_clipped_stats(peaks, mask=mask, sigma=2.5)
    peaks = fabada(peaks,3*(std**2),verbose=verbose)
    if mask is not None: peaks[np.where(mask != 0)] = np.nan
    peaks[peaks==0] = np.nan
    return peaks



def sharp_mask(data, C=7, W=5, mask=False, enhace=True):
    '''
    Contrast Enhancement using Digital Unsharp Masking
    https://www.astronomy.ohio-state.edu/pogge.1/Ast350/Unsharp/
    '''
    original_stats = sigma_clipped_stats(data, mask=mask, sigma=2.5)
    if enhace: enhaced = fabada(data, 2*(original_stats[2]**2),verbose=True)
    else: enhaced = data
    smooth = gaussian_filter(data,sigma=W)
    return C*enhaced - (C-1)*smooth

def get_bipeaks(data,n=2.5,sigma=5,verbose=False):

    original_stats = sigma_clipped_stats(data, sigma=2.5)
    recover = fabada(data, 2*(original_stats[2]**2),verbose=verbose)
    peaks = recover - gaussian_filter(data, sigma=sigma)
    residual_stats = sigma_clipped_stats(peaks, sigma=2.5)
    peaks = fabada(peaks,3*(residual_stats[2]**2),verbose=verbose)
    
    peaks[peaks > n*residual_stats[2]] *= 255
    peaks[peaks < n*residual_stats[2]] = 0
    return peaks

def increase_mask(mask, shape=(3,3)):
    import cv2
    if mask.dtype != np.int16: mask = mask.astype(np.int16)
    return cv2.dilate(mask,np.ones(shape)/np.prod(shape), iterations=1)


def sexmask(IMG, folders, fwhm=None, plot=False, temp=False):
    '''
    Use SExtractor to create a mask of the image
    We run FABADA to enhace the object and improve the detection.
    It has three steps:
        1. Run SExtractor with default parameters to detect objects
        2. Run SExtractor with default parameters to detect point sources
        3. Run SExtractor in the residual image to detect point sources inside the galaxy
    The last two steps has a gaussian weight distance and area filter to avoid the masking 
    of extended inner regions of the galaxy.
    TODO: in (3) add filter of detected objects and only apply to nxReff of galaxy


    Parameters
    ----------
        IMG : astropipe.classes.Image object
            Class object containing the image data

        folders : astropipe.classes.Directories object
            Class object containing the directories of the project

        fwhm : float, optional
            FWHM of the image in pixels. It is used to filter point sources.
            if None, it measures it from SExtractor.

        plot : bool, optional
            If True, plot the mask and the image and save it in temporary directory

    Returns
    -------
        bool : True if the mask is created
    '''
    import numpy as np
    from astropy.io import fits
    from matplotlib import pyplot as plt


    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.name).split(IMG.extension)[0]+'_mask.fits'))
    if hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0

    back_size = 8*fwhm**2 if fwhm else 120

    if not hasattr(IMG,'std'): 
        mean, bkg, IMG.std = sigma_clipped_stats(IMG.data, sigma=2.5)

    # Step 1: Run FABADA to smooth image an improve detection and initialize mask

    recover = fabada(IMG.data,2*(IMG.std**2),verbose=True)
    recover[np.isnan(IMG.data)] = np.nan
    IMG.set_mask(np.zeros_like(IMG.data))

    # Step 2: Run SExtractor
    #   Step 2.1: Run SExtractor with default param with 1.3 sigma threshold

    params = ['ISOAREA_IMAGE','ELLIPTICITY','FWHM_IMAGE', 'THETA_IMAGE']

    defaultsex_config  = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                                    'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                                        IMG.name+'_sex.fits'),
                                    'PIXEL_SCALE' : IMG.pixel_scale,
                                    'DETECT_THRESH': 1.3,   # 1.5
                                    "DEBLEND_MINCONT": 0.005,
                                    "DEBLEND_NTHRESH": 32,
                                    'PHOT_FLUXFRAC': 0.9, 
                                    'BACK_SIZE':  back_size,       # 64
                                    'MEMORY_OBJSTACK': 10000,           
                                    'MEMORY_PIXSTACK': 1000000,     
                                    'MEMORY_BUFSIZE': 51200}

    defaultsex = SExtractor(config= defaultsex_config, params=params)
    defaultsex.run(recover)
    IMG.obj_id(defaultsex)

    if fwhm is None: 
        stars = defaultsex.catalog['CLASS_STAR']==1.0
        if np.sum(stars) == 0: stars = defaultsex.catalog['CLASS_STAR'] > 0.8
        fwhm = np.int8(np.nanmean(defaultsex.catalog['FWHM_IMAGE'][stars]))

    object_index = np.where(defaultsex.objects == IMG.id)
    defaultsex.objects[object_index] = 0

    if not hasattr(IMG,'r_eff'): IMG.r_eff = float(defaultsex.catalog[IMG.id-1]['FLUX_RADIUS'])
    if not hasattr(IMG,'eps'):   IMG.eps = float(defaultsex.catalog[IMG.id-1]['ELLIPTICITY'])
    if not hasattr(IMG,'pa'):    IMG.pa = float(defaultsex.catalog[IMG.id-1]['THETA_IMAGE'])

    index = filter_distance2D(np.array([defaultsex.catalog['X_IMAGE'],
                                    defaultsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff)
    for ind in sigma_filter(defaultsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE']):
        if defaultsex.catalog['CLASS_STAR'][ind-1] < 0.5:
            defaultsex.objects[np.where(defaultsex.objects == ind)] = 0
    
    IMG.data.mask[np.where(defaultsex.objects != 0)] = 1

    #   Step 2.2: Run SExtractor for point sources

    masked_image = np.zeros_like(recover) + recover
    masked_image[IMG.data.mask] = np.nan
    masked_image[object_index] = np.nan

    back_size = 3*fwhm**2
    midsex_config  = defaultsex_config.copy()
    midsex_config['DETECT_THRESH'] = 1.5
    midsex_config['BACK_SIZE'] = back_size
    
    midsex = SExtractor(config= midsex_config, params=params)
    midsex.run(masked_image)
    IMG.obj_id(midsex)
    midsex.objects[np.where(midsex.objects == IMG.id)] = 0

    weights = weight_distance2D(np.array([midsex.catalog['X_IMAGE'],
                                          midsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([midsex.catalog['X_IMAGE'],
                                        midsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(midsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE'],
                        sigma=3, weights=weights[index]):
        if midsex.catalog['CLASS_STAR'][ind-1] < 0.8:
            midsex.objects[np.where(midsex.objects == ind)] = 0

    midsex.objects = increase_mask(midsex.objects,shape=(3,3))

    IMG.data.mask[np.where(midsex.objects != 0)] = 1

    #   Step 2.3: Run SExtractor residuals from gaussian smoothing

    pointsex_config = midsex_config.copy()
    pointsex_config['DETECT_THRESH'] = 1.0

    pointsex = SExtractor(config = pointsex_config, params=params)
    pointsex.run(get_bipeaks(IMG.data.data,n=2.5,sigma=5))
    IMG.obj_id(pointsex)
    pointsex.objects[np.where(pointsex.objects == IMG.id)] = 0

    weights = 1.5*weight_distance2D(np.array([pointsex.catalog['X_IMAGE'],
                                              pointsex.catalog['Y_IMAGE']]),
                            IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([pointsex.catalog['X_IMAGE'],
                                        pointsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)
    index = sigma_filter(pointsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE'], 
                            sigma=3, weights=weights[index])

    for ind in sigma_filter(pointsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE'], 
                        sigma=3, weights=weights[index]):
        pointsex.objects[np.where(pointsex.objects == ind)] = 0

    
    IMG.data.mask[np.where(pointsex.objects != 0)] = 1

    # Filter mask and growth math according to FWHM

    kernel = gaussian2D(l=fwhm,sig=fwhm//2)
    IMG.data.mask  = cv2.dilate(cv2.erode(IMG.data.mask.astype(np.float32),kernel),kernel)

    IMG.data.mask = cv2.dilate(IMG.data.mask.astype(np.float32), kernel)


    #   Step 3: Save the files 
    
    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header
                    ).writeto(os.path.join(folders.temp,f'{IMG.name}_masked.fits')
                    ,overwrite=True)

    mask_array = np.array(IMG.data.mask, dtype = np.uint8)
    fits.PrimaryHDU(mask_array, header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: # Save a plotting result with the image and the mask
        IMG.show(width=300)
        plt.savefig(os.path.join(folders.out,IMG.name+'_mask.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    if not temp:  # Remove temporary files
        os.remove(os.path.join(folders.temp,f'{IMG.name}_masked.fits'))
        os.remove(os.path.join(folders.temp,f'{IMG.name}_sex.fits'))
    
    return os.path.isfile(folders.mask)

def fastmask(data, center, nsigma=1, fwhm=5):
    '''
    create quick mask for testing porpuses
    Parameters:
    ----------
        data : 2D array
                Image data.
        center : tuple
                Center of the galaxy (x,y).
        nsigma : float
                Number of sigma to use as threshold. Default is 1.
        fwhm : float
                Full width half maximum of the gaussian kernel. Default is 5.
    Returns:
    -------
        mask : 2D array
                Mask with the same shape as data.
    '''
    x,y = center
    center = (np.int16(y),np.int16(x))
    mask = np.zeros_like(data)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)    
    mask[data > median + nsigma*std] = 1
    mask = cv2.connectedComponentsWithAlgorithm(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_WU)[1]
    mask[mask==mask[center]]=0
    kernel = np.ones((fwhm//2 +1,fwhm//2 +1))
    mask =  cv2.dilate(cv2.erode(mask.astype(np.float32),kernel),kernel)
    peaks = get_bipeaks(data)
    N,label,stats,_ = cv2.connectedComponentsWithStatsWithAlgorithm(peaks.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_WU)
    eps = stats[:,2]/stats[:,3]
    label[np.isin(label, np.argwhere((eps<0.8) + (eps>1.2)))]=0
    mask[label!=0] = 1
    mask = cv2.connectedComponentsWithAlgorithm(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_WU)[1]
    mask[center]=0
    mask = cv2.dilate(mask.astype(np.float32),kernel).astype(np.uint16)
    return mask

def mtomask(IMG, folders, plot=False):
    '''
    Use MTObjects to create a mask of the image
    We run FABADA to enhace the object and improve the detection.
    It has three steps:
        1. Run MTObjects with default parameters to detect objects
        2. Run MTObjects with default parameters to detect point sources
        3. Run MTObjects in the residual image to detec point sources inside the galaxy
    The last two steps has a gaussian weight distance and area filter to avoid the masking 
    of extended inner regions of the galaxy.
    TODO: in (3) add filter of detected objects and only apply to nxReff of galaxy


    Parameters
    ----------
        IMG : astropipe Image object
            Class object containing the image data

        folders : astropipe Directories object
            Class object containing the directories of the project

        plot : bool, optional
            If True, plot the mask and the image and save it in temporary directory

    Returns
    -------
        bool : True if the mask is created
    '''

    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.file).split(IMG.extension)[0]+'_mask.fits'))
    if hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0

    if not hasattr(IMG,'std'): 
        mean, bkg, IMG.std = sigma_clipped_stats(IMG.data, sigma=2.5)

    
    # Step 1: Run FABADA to smooth image an improve detection and initialize mask

    recover = fabada(IMG.data,2*(IMG.std**2),verbose=True)
    recover[np.isnan(IMG.data)] = np.nan
    IMG.set_mask(np.zeros_like(IMG.data))

    # Step 2: Run SExtractor
    #   Step 2.1: Run SExtractor with default param with 1.3 sigma threshold

    params = ['ISOAREA_IMAGE','ELLIPTICITY','FWHM_IMAGE', 'THETA_IMAGE']

    defaultsex_config  = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                                    'CHECKIMAGE_NAME' : os.path.join(folders.temp,
                                                        IMG.name+'_sex.fits'),
                                    'PIXEL_SCALE' : IMG.pixel_scale,
                                    'DETECT_THRESH': 1.0,   # 1.5
                                    "DEBLEND_MINCONT": 0.005,
                                    "DEBLEND_NTHRESH": 32,
                                    'PHOT_FLUXFRAC': 0.9, 
                                    'BACK_SIZE':  120,       # 64
                                    'MEMORY_OBJSTACK': 10000,           
                                    'MEMORY_PIXSTACK': 1000000,     
                                    'MEMORY_BUFSIZE': 51200}

    defaultsex = SExtractor(config= defaultsex_config, params=params)
    defaultsex.run(recover)
    IMG.obj_id(defaultsex)

    defaultsex.objects[np.where(defaultsex.objects == IMG.id)] = 0
    
    if not hasattr(IMG,'r_eff'): IMG.r_eff = float(defaultsex.catalog[IMG.id-1]['FLUX_RADIUS'])
    if not hasattr(IMG,'eps'):   IMG.eps = float(defaultsex.catalog[IMG.id-1]['ELLIPTICITY'])
    if not hasattr(IMG,'pa'):    IMG.pa = float(defaultsex.catalog[IMG.id-1]['THETA_IMAGE'])

    index = filter_distance2D(np.array([defaultsex.catalog['X_IMAGE'],
                                    defaultsex.catalog['Y_IMAGE']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff)
    for ind in sigma_filter(defaultsex.catalog[index], ['ISOAREA_IMAGE','FWHM_IMAGE']):
        if defaultsex.catalog['CLASS_STAR'][ind-1] < 0.5:
            defaultsex.objects[np.where(defaultsex.objects == ind)] = 0
    
    defaultsex.objects = increase_mask(defaultsex.objects,shape=(4,4))
    IMG.data.mask[np.where(defaultsex.objects != 0)] = 1

    #   Step 2.2: Run MTObjects 

    mto = MTObjects()
    mto.move_factor = 0.01 # To imporve detection of faint outskirts of objects
    
    mto.run(recover)
    IMG.obj_id(mto)
    mto.objects[np.where(mto.objects == IMG.id)] = 0

    weights = weight_distance2D(np.array([mto.catalog['X'],
                                          mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([mto.catalog['X'],
                                        mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(mto.catalog[index], ['area','R_fwhm'],
                        sigma=3, weights=weights[index], colid='ID'):
        mto.objects[np.where(mto.objects == ind)] = 0

    mto.objects = increase_mask(mto.objects,shape=(3,3))

    IMG.data.mask[np.where(mto.objects != 0)] = 1

    #   Step 2.3: Run MTObjects in residuals from gaussian smoothing

    mto.run(get_bipeaks(IMG.data.data,n=2.5,sigma=5))
    IMG.obj_id(mto)
    mto.objects[np.where(mto.objects == IMG.id)] = 0

    weights = 1.5*weight_distance2D(np.array([mto.catalog['X'],
                                              mto.catalog['Y']]),
                            IMG.pix, IMG.pa*np.pi/180, IMG.eps, 2*IMG.r_eff/3)

    index = filter_distance2D(np.array([mto.catalog['X'],
                                        mto.catalog['Y']]),
                           IMG.pix, IMG.pa*np.pi/180, IMG.eps, 1.5*IMG.r_eff)

    for ind in sigma_filter(mto.catalog[index], ['area','R_fwhm'], 
                            sigma=3, weights=weights[index],colid='ID'):
        mto.objects[np.where(mto.objects == ind)] = 0

    mto.objects = increase_mask(mto.objects, shape = (2,2))

    IMG.data.mask[np.where(mto.objects != 0)] = 1

    #   Step 3: Save the files 
    
    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header
                    ).writeto(os.path.join(folders.temp,f'{IMG.name}_masked.fits')
                    ,overwrite=True)

    mask_array = np.array(IMG.data.mask, dtype = np.uint8)
    fits.PrimaryHDU(mask_array, header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: 
        IMG.show(width=300)
        plt.savefig(os.path.join(folders.out,IMG.name+'_mask.jpg'), dpi=300, bbox_inches='tight', pad_inches=0.1)

    return os.path.isfile(folders.mask)

def automatic_mask(IMG,folders,sex_config=point_sexcofing,nc_config='--numthreads=8',plot=True):


    if not hasattr(folders,'mask'): folders.set_mask(os.path.join(folders.out,os.path.basename(IMG.file).split(IMG.extension)[0]+'_mask.fits'))
    if hasattr(IMG,'bkg'): bkg = IMG.bkg
    else: bkg = 0
    gnu = AstroGNU(IMG.file, hdu=IMG.hdu, dir=folders.temp)
    gnu.noisechisel(config=nc_config, keep=True)
    gnu.segment(config=nc_config, keep=True)
    gnu.make_catalog(zp=IMG.zp, config=nc_config)
    
    IMG.noise(gnu.objects,
            plot=os.path.join(folders.temp,IMG.name+"_noise.png"))

    if not "CHECKIMAGE_NAME" in sex_config:
         sex_config["CHECKIMAGE_NAME"] =  os.path.join(folders.temp,
                                    IMG.name+'_sex.fits')
    
    if not 'PIXEL_SCALE' in sex_config:
        sex_config['PIXEL_SCALE'] = IMG.pixel_scale     

    recover = fabada(IMG.data,3*IMG.std,verbose=True)
    ast_obj = detect_sources(recover - bkg, 5*IMG.std, npixels=5)
    
    sex = SExtractor(config = sex_config)
    
    sex.run(recover)

    # Masking Process
    # Step 1: Initialize Image and Masks

    IMG.obj_id(gnu)
    IMG.obj_id(sex)
    ast_id = ast_obj.data[np.int64(IMG.pix[1]),np.int64(IMG.pix[0])]

    # Step 2: Create final Mask (True(1) = masked ; False(0) = Not-masked)

    IMG.set_mask(np.zeros_like(IMG.data))

    #   Step 2.1: Mask all objects of SExtractor outside of Segment source

    IMG.data.mask[where([sex.objects != IMG.id,
               gnu.objects != IMG.id,
               sex.objects != 0])] = 1

    #   Step 2.2: Mask all objects of SExtractor outside of Astropy source

    IMG.data.mask[where([ast_obj.data != ast_id,
                         sex.objects != 0])] = 1

    
    #   Step 2.3: Mask all objects of SExtractor that overlaps one pixel 
    #             with Astropy source

    overlap = np.unique(sex.objects[where([ast_obj.data == ast_id, sex.objects != 0])])
    for id in overlap:
        IMG.data.mask[np.where(sex.objects==id)]=0

    IMG.set_catalog(gnu.catalog)


    # Mask faint extended regions 

    temporary_masked_file = os.path.join(folders.temp,f'{IMG.name}_mask_point.fits')

    masked = IMG.data.data.copy()
    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header).writeto(temporary_masked_file,overwrite=True)


    gnu_extended = AstroGNU(temporary_masked_file, hdu=0, dir=folders.temp)

    gnu_extended.noisechisel(config=nc_config+' --tilesize=7,7', keep=True)
    gnu_extended.segment(config='--gthresh=1 --snminarea=30 ', keep=True)

    id_extended = gnu_extended.objects[np.int(IMG.pix[1]),np.int(IMG.pix[0])]

    IMG.data.mask[where([gnu_extended.objects != id_extended,
                         gnu_extended.objects != 0])] = 1
    

    # Mask again with SExtractor for point sources using the applied mask so far

    masked[np.where(IMG.data.mask==1)] = np.nan

    sex.run(masked)
    IMG.obj_id(sex)
    sex.objects[sex.objects == IMG.id] = 0
    # sex.objects = cv2.dilate(sex.objects.astype(np.int16), gaussian(3,(10, 10)), iterations=1)
    sex.objects = convolve2d(sex.objects, gaussian(3,(10, 10)), mode='same', boundary='fill', fillvalue=0)
    IMG.data.mask[sex.objects != 0] = 1



    # Mask interior regions of extended sources

    peaks = binarize_peaks(IMG.data.data,IMG.data.mask,theshold=8, sigma=5)
    sex.run(peaks)
    mean, median, std = sigma_clipped_stats(sex.catalog['FLUX_RADIUS'])
    bigregions = sex.catalog['NUMBER'][sex.catalog['FLUX_RADIUS'] > median+5*std]
    IMG.obj_id(sex)
    sex.objects[sex.objects==IMG.sex_id] = 0
    for biggie in bigregions:
        sex.objects[sex.objects == biggie] = 0
    sex.objects = convolve2d(sex.objects, gaussian(2,(6, 6)), mode='same', boundary='fill', fillvalue=0)
    IMG.data.mask[sex.objects != 0] = 1


    # Save the files 

    masked[np.where(IMG.data.mask==1)] = np.nan
    fits.PrimaryHDU(masked,header=IMG.header).writeto(temporary_masked_file,overwrite=True)

    mask_array = np.array(1-IMG.data.mask,dtype = np.uint8)
    fits.PrimaryHDU(mask_array,header=IMG.header).writeto(folders.mask,overwrite=True)

    if plot: 
        IMG.show()
        plt.savefig(os.path.join(folders.temp,IMG.name+'_mask.jpg'),dpi=300)
    gnu.remove()
    gnu_extended.remove()

def ds9_region_masking(IMG,folders):

    file = os.path.join(folders.regions,IMG.name+'.reg')

    df_region = pd.read_csv(file,delimiter=' ',header=None,
                        names=['frame', 'region','ra','dec','radius'])
    
    
    if ':' in df_region['ra'][0]:
        units = (u.hourangle,u.deg)
    else:
        units = (u.deg,u.deg)
    
    for index, row in df_region.iterrows():

        pixel_center = skycoord_to_pixel(SkyCoord(row['ra'],row['dec'], unit=units,frame="fk5"), IMG.wcs)
        pixel_radius = np.float64(row['radius'][:-1])/IMG.pixel_scale
        aperture = CircularAperture(pixel_center, pixel_radius)
        mask_region = aperture.to_mask(method='exact')
        IMG.data.mask[np.where(mask_region.to_image(IMG.data.mask.shape)==1)] = True
    
    mask_array = np.array(1-IMG.data.mask,dtype = np.uint8)
    fits.PrimaryHDU(mask_array,header=IMG.header).writeto(folders.mask,overwrite=True)

    return True

