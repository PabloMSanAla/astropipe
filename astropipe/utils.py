import os
import sys
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import cv2
import fnmatch
from copy import deepcopy
from fabada import fabada
import argparse
from math import gamma
from PyPDF2 import PdfReader, PdfMerger


from astropy.wcs import WCS, utils
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.signal import convolve2d

from scipy.signal import argrelextrema
from scipy import stats

from photutils.centroids import centroid_com, centroid_quadratic, centroid_2dg

from autoprof.pipeline_steps import Center_HillClimb

import datetime
import warnings


def mag_limit(std, Zp=22.5, omega=10, scale=0.33, n=3):
    ''' Computes the surface brightness limit of the image 
    given the STD according to Román et al. 2020. Appendix A. 
    
    Parameters
    ----------
        std : float, or array
            Standard deviation of the image.
        Zp : float, optional
            Zero point of the image. The default is 22.5.
        omega : float, optional
            Area of the image in arcsec^2. The default is 10.
        scale : float, optional
            Scale of the image in arcsec/pixel. The default is 0.33.
        n : int, optional
            Number of sigma to use. The default is 3.
    
    Returns
    -------
        mag : float
            Surface brightness limit of the image.
    '''
    return -2.5 * np.log10(n * std / (scale * omega)) + Zp

def redshift_to_kpc(redshift,H0=70,Tcmb0 = 2.725, Om0=0.3):
    '''Function that given a redshift returns the physical distance
    in kpc. It uses the cosmology from astropy.cosmology.
    
    Parameters
    ----------
        redshift : float, or array
            Redshift of the object.
        H0 : float, optional
            Hubble constant. The default is 70. [km/s/Mpc]
        Tcmb0 : float, optional
            CMB temperature. The default is 2.725. [K]
        Om0 : float, optional
            Matter density. The default is 0.3.
    
    Returns
    -------
        distance : float
            Physical distance in kpc.
    '''
    cosmo = FlatLambdaCDM(H0=H0* u.km / u.s / u.Mpc, Tcmb0=Tcmb0* u.K , Om0=Om0)
    return (cosmo.luminosity_distance(redshift) * 1000 * u.kpc/u.Mpc)

def redshift_to_gyr(z, H0=70, Tcmb0=2.725, Om0=0.3):
    """
    Converts redshift to Gyr using LambdaCDM cosmology.

    Parameters
    ----------
        redshift : float, or array
            Redshift of the object.
        H0 : float, optional
            Hubble constant. The default is 70. [km/s/Mpc]
        Tcmb0 : float, optional
            CMB temperature. The default is 2.725. [K]
        Om0 : float, optional
            Matter density. The default is 0.3.
    
    Returns
    -------
        Age : float
            Age of the Universe in Gyr
    """
    cosmo = FlatLambdaCDM(H0=H0* u.km / u.s / u.Mpc, Tcmb0=Tcmb0* u.K , Om0=Om0)
    age = cosmo.age(z)
    return age.to(u.Gyr).value

def kpc_to_arcsec(kpc, redshift, H0=70, Tcmb0=2.725, Om0=0.3):
    '''Function that given a physical size of an object
    and its redshift it converts it into the
    angular size in arcseconds.
    
    Parameters
    ----------
        kpc : float
            Physical size in kpc.
        redshift : float
            Redshift of the object.
        H0 : float, optional
            Hubble constant. The default is 70. [km/s/Mpc]  
        Tcmb0 : float, optional
            CMB temperature. The default is 2.725. [K]
        Om0 : float, optional
            Matter density. The default is 0.3.
    
    Returns
    -------
        arcsec : float
            Angular size in arcseconds.
    '''
    cosmo = FlatLambdaCDM(H0=H0* u.km / u.s / u.Mpc, Tcmb0=Tcmb0* u.K , Om0=Om0)
    da = cosmo.angular_diameter_distance(redshift)*1e3  # Mpc --> kpc
    arcsec_to_rad = np.pi/(180*3600)
    return (kpc)/(arcsec_to_rad*da)

def arcsec_to_kpc(arcsec, redshift, H0=70, Tcmb0=2.725, Om0=0.3):
    '''Function that given a projected angular size of an object
    and its redshift it converts it into the
    physical size in same units. 

    Parameters
    ----------
        arcsec : float
            Angular size in arcsec.
        redshift : float
            Redshift of the object.
        H0 : float, optional
            Hubble constant. The default is 70. [km/s/Mpc]
        Tcmb0 : float, optional
            CMB temperature. The default is 2.725. [K]
        Om0 : float, optional
            Matter density. The default is 0.3.
    
    Returns
    -------
        kpc : float
            Physical size in kpc.
      '''
    cosmo = FlatLambdaCDM(H0=H0* u.km / u.s / u.Mpc, Tcmb0=Tcmb0* u.K , Om0=Om0)
    da = cosmo.angular_diameter_distance(redshift)*1e3  # Mpc --> kpc
    arcsec_to_rad = np.pi/(180*3600)
    return (arcsec*arcsec_to_rad*da).value

def convert_PA(angle):
    if angle <0:
        return 180 + angle
    else:
        return angle

def where(list_conditions):
    index = list_conditions[0]
    for cond in list_conditions[1:]:
        index = np.multiply(index, cond)
    return np.where(index)

def binarize(image, nsigma=1, mask=None, center=None):
    """Binarize an image using a threshold of nsigma*std.
    Statistics are computed with sigma clipped, then the 
    image binarize is dilate to remove noise. 

    Parameters
    ----------
        image : array_like
            Image to binarize.
        nsigma : float, optional
            Number of sigma to use as threshold.
        mask : array_like, optional
            If use, mask is applied.
        center: tuple, optional
            Center of the object to binarize. (x,y)
    
    Returns
    -------
        binary : array_like
            Binarized image.
    """
    if center is None:
        y,x = np.array(image.shape)//2
    else:
        x,y = np.int64(center)
    
    mask = None if mask is None else mask
    if hasattr(image, 'mask'): 
        if mask is None: mask = np.ma.getmask(image)
        image = np.ma.getdata(image)

    _, median, std = sigma_clipped_stats(image, sigma=2.5, mask=mask, maxiters=2)
    smooth = fabada(image, std**2)
    _, median, std = sigma_clipped_stats(image, sigma=2.5, mask=mask, maxiters=2)
    index = smooth > median+nsigma*std
    binarized = np.zeros_like(image)
    binarized[index * ~mask] = 1
    label = cv2.connectedComponentsWithAlgorithm(binarized.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S, ccltype=cv2.CCL_WU)[1]
    binarized[label != label[y,x]] = 0
    binarized = cv2.erode(binarized, np.ones((5,5)), iterations=1)
    return binarized



def morphology(binary):
    ''' Gets the morphological parameters of a 
    binary image using the image moments. 
    
    Parameters
    ----------
        binary : array_like
            Binary image.
    
    Returns
    -------
        angle : float 
            Angle of the object in degrees [x to y]
        sma : float
            Semi-major axis of the object in pixels
        eps : float
            Ellipticity of the object [1-b/a]
    # '''

    moments = cv2.moments(binary)
    
    x2 = (moments['m20']/moments['m00'])-(moments['m10']/moments['m00'])**2
    y2 = (moments['m02']/moments['m00'])-(moments['m01']/moments['m00'])**2
    xy = (moments['m11']/moments['m00'])-(
                moments['m10']*moments['m01']/moments['m00']**2)
    
    major = np.sqrt(0.5*(x2+y2) + np.sqrt((0.5*(x2-y2))**2 + xy**2))
    minor = np.sqrt(0.5*(x2+y2) - np.sqrt((0.5*(x2-y2))**2 + xy**2))
    angle = 0.5*np.arctan2(2*xy,x2-y2)*180/np.pi
    eps = 1 - minor/major

    arguments = np.argwhere(binary)
    xdist = np.nanmax(arguments[:,1]) -  np.nanmin(arguments[:,1])
    ydist = np.nanmax(arguments[:,0]) -  np.nanmin(arguments[:,0]) 
    sma_x = xdist / np.cos(angle*np.pi/180)
    sma_y = ydist / np.sin(angle*np.pi/180)
    sma = np.nanmax([sma_x,sma_y])/2

    return angle,sma,eps

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)
            

def run_fabada(IMG,std=None, verbose=True,max_iter=3000):
    if not std: std=IMG.std
    recover = deepcopy(IMG)
    if hasattr(IMG.data, 'mask'):
        recover.data = fabada(recover.data.data,std**2,max_iter=max_iter,verbose=verbose)
        recover.data = np.ma.masked_array(recover.data,mask=IMG.data.mask)
    else:
        recover.data = fabada(recover.data,std**2,max_iter=max_iter,verbose=verbose)
    
    recover.name = recover.name+'_fabada'
    return recover
            

def make_parser():
    """Create an argument parser taken from MTObjects."""
    parser = argparse.ArgumentParser(description="Index of csv file")

    parser.add_argument("-index", type=int, help="Index", default=0)

    return parser

def closest(data, value):
    '''
    Find a value in an array closest to the input value

    Parameters
    ----------
        array : numpy.array
            Input 1D array
        value : float
            Value in array you want to find
    
    Returns
    -------
        index : int
            Index of the closest value in the array
    '''
    return np.nanargmin(np.abs(data-value))



def cutout(file, center, width, hdu=0, mode='image', out=None):


    '''
    Crop a fits file to a given width and center preserving the WCS. 
    
    Parameters
    ----------
        file: str
            Fits file name.
        center: tuple, optional
            Center of the cropped image in pixels (image) or in degrees (wcs).
        width: tuple
            Width of the cropped image in pixels (image) or in degrees (wcs).
        hdu: int, optional
            HDU to crop. Default is 0.
        mode: str, optional
            'image' or 'wcs'.
        out: str, optional
            Output file name. Default is filename_crop.fits.

    Returns
    -------
        out: str
            Cropped fits file name.
            yes?
    '''
    
    data = fits.getdata(file,hdu)
    header = fits.getheader(file,hdu)
    wcs = WCS(header)

    if mode == 'wcs':
        pixel_scales = utils.proj_plane_pixel_scales(wcs)
        width = np.int64([width[1] / pixel_scales[1], width[0]/pixel_scales[0]])
        center = wcs.world_to_pixel_values(*center)
    elif mode == 'image':
        center = np.int64(center)
        width = np.int64(width)[::-1]
    else:
        raise ValueError('mode must be image or wcs')

    # cropped, header, coords = crop(data, header, center, width)
    cropped = Cutout2D(data, center, width, wcs=wcs)
    
    header.update(cropped.wcs.to_header())
    cutout_hdu = fits.PrimaryHDU(data=cropped.data, header=header)

    if out is None: out = file.replace('.fits','_crop.fits')
    
    cutout_hdu.header['COMMENT'] = "= Cropped fits file ({}).".format(datetime.date.today())
    cutout_hdu.header['CENTER'] = (f'{center[0]:.2f},{center[1]:.2f}',
                                   'Center of the cutout [x,y]')
    cutout_hdu.header['SIZE'] = (f'{width[0]:d},{width[1]:d}',
                                 'Size of the cutout [x,y]')
    cutout_hdu.header['ORFILE'] = (os.path.basename(file), 'Original file')

    cutout_hdu.writeto(out, overwrite=True)

    return os.path.isfile(out)


def center_hillclimb(data, center, centeringring = 10, pixscale = 0.1):
    '''
    Finds the center of an object in an image using a Hill Climb algorithm
    Implementation from astropipe.pipeline_steps.Center_HillClimb.

    Parameters
    ----------
        data : array_like
            Image data.
        center : tuple
            Initial center of the object in the image. (x,y)
        centeringring : int, optional
            Radius of the centering ring. AutoProf parameter. Default is 10. 
        pixscale : float, optional
            Pixel scale of the image. Default is 0.1.

    Returns
    -------
        x,y : tuple
            New center of the object in the image.
    '''

    options = {'ap_name':'','ap_centeringring':centeringring, 'ap_pixscale':pixscale,
                'ap_guess_center':  {'x':center[0],'y':center[1]}}
    results = {'background': 0.0, 'background noise': 0.0, 'psf fwhm': 1}

    _,new_results = Center_HillClimb(data, results, options)

    return new_results['center']['x'],new_results['center']['y']

def averageBinning(data, binning=2):
    """
    Bin pixels in a 2D array using the average of the values of the bin
    Parameters:
    - data: 2D numpy array of data values
    - binning: Size of the bins (integer). Default = 2

    Returns:
    - binned_data: 2D numpy array of binned data values
    - binned_mask: 2D numpy array of updated mask values after binning
    """

    if hasattr(data,'mask'):
        mask = data.mask
    else:
        mask = np.zeros_like(data)
    # Determine the shape of the binned data
    new_shape = (data.shape[0] // binning, data.shape[1] // binning)

    # Reshape the data and mask arrays into 2D arrays of equal-sized bins
    binned_data = data[:new_shape[0] * binning, :new_shape[1] * binning].reshape(new_shape[0], binning, new_shape[1], binning).mean(axis=(1, 3))
    binned_mask = mask[:new_shape[0] * binning, :new_shape[1] * binning].reshape(new_shape[0], binning, new_shape[1], binning).any(axis=(1, 3))

    binned = np.ma.masked_array(binned_data,mask=binned_mask)
    return binned

def sumBinning(data, binning=2):
    """
    Bin pixels in a 2D array while preserving total flux, and update the mask accordingly.

    Parameters:
    - data: 2D numpy array of data values
    - binning: Size of the bins (integer). Default = 2

    Returns:
    - binned_data: 2D numpy array of binned data values
    - binned_mask: 2D numpy array of updated mask values after binning
    """

    if hasattr(data,'mask'):
        mask = data.mask
    else:
        mask = np.zeros_like(data)
    # Determine the shape of the binned data
    new_shape = (data.shape[0] // binning, data.shape[1] // binning)

    # Reshape the data and mask arrays into 2D arrays of equal-sized bins
    binned_data = data[:new_shape[0] * binning, :new_shape[1] * binning].reshape(new_shape[0], binning, new_shape[1], binning).mean(axis=(1, 3))
    binned_mask = mask[:new_shape[0] * binning, :new_shape[1] * binning].reshape(new_shape[0], binning, new_shape[1], binning).any(axis=(1, 3))

    binned = np.ma.masked_array(binned_data,mask=binned_mask)
    return binned

def get_pixel_scale(header):
    '''
    Funtion that given a header returns the pixel scale in arcsec/pixel.

    PARAMETERS
    ----------
        header: astropy.io.fits.header.Header
            astropy header class of the image
    
    RETURNS
    -------
        scale: float
            median pixel scale of the pixel in arcseconds

    '''
    wcs = WCS(header)
    scale = np.abs(np.mean(utils.proj_plane_pixel_scales(wcs))*3600)
    return scale

def crop(data, header, center, width, out=None):
    '''
    Funtion that crops a image given a center and a width updating the WCS
    information in the header.

    Parameters
    ----------
        data: array_like
            Image to crop.
        header: astropy.io.fits.header.Header
            Header of the image.
        center: tuple
            Center of the cropped image in pixels.
        width: tuple
            Width of the cropped image in pixels. (x,y)
    
    Returns
    -------
        new_data: array_like
            Cropped image.
        new_header: astropy.io.fits.header.Header
            Header of the cropped image.    
    '''
    wcs = WCS(header)
    center = np.int32(center)
    width = np.int32(width)
    hasmask = hasattr(data, 'mask')

    if hasmask: data , mask = data.data , data.mask

    # Set limits of the cropping
    x0 , x1 = np.int64((center[0]-(width[0]//2 + 1),center[0]+width[0]//2))
    y0 , y1 = np.int64((center[1]-(width[1]//2 + 1),center[1]+width[1]//2))
    
    # Sanity checks
    if x0 < 0: x0 = 0
    if x1 > data.shape[1]: x1 = data.shape[1]
    if y0 < 0: y0 = 0
    if y1 > data.shape[0]: y1 = data.shape[0]
    
    # Crop image and wcs
    new_data = data[y0:y1, x0:x1]
    new_wcs = wcs[y0:y1, x0:x1]

    if hasmask:
        new_mask = mask[y0:y1, x0:x1]
        new_data = np.ma.masked_array(new_data, mask=new_mask)
    
    header.update(new_wcs.to_header())
    header['COMMENT'] = "= Cropped fits file ({}).".format(datetime.date.today())
    header['ICF1PIX'] = (f'{y0}:{y1},{x0}:{x1}',
                          'Range of pixels used for this cutout [y0:y1,x0:x1]')
    
    if out is not None: fits.PrimaryHDU(new_data, header).writeto(out, overwrite=True)
    
    return new_data, header, ((x0,x1),(y0,y1))

def localSlope(x, y):
    '''
    For deriving the local slope profile of two arrays

    Parameters
    ----------
    x : numpy.ndarray
        The abscissa axis array
    y : numpy.ndarray
        The ordinate axis array

    Returns
    -------
    dydx : numpy.ndarray
        An array of slope values between x and y
    '''
    dydx = np.zeros(len(y))
    dydx[1:] = np.diff(y) / np.diff(x)
    dydx[0] = y[0] / x[0]  # Replace first value with ratio of initial points

    return dydx

def derivative(x, y, n=4):
    """
    Computes de slope from the n adjacent points using 
    linear regression.

    Parameters
    ----------
        x : array
            x-axis values.
        y : array
            y-axis values.
        n : int, optional
            Number of adjacent points to use. The default is 4.
    
    Returns
    -------
        deriv : array
            Slope (dy/dx) of the line x and y.
    """
    index = np.isfinite(x) & np.isfinite(y)
    deriv = np.zeros_like(x)
    x = np.array(x)[index]
    y = np.array(y)[index]
    
    der = np.zeros_like(x)
    for i in range(len(x)):
        if i<n:
            slope = stats.linregress(x[:i+n],y[:i+n])[0]
        elif len(x)-i<n:
            slope = stats.linregress(x[i-n:],y[i-n:])[0]
        else:
            slope = stats.linregress(x[i-n:i+n],y[i-n:i+n])[0]
        der[i] = slope
    deriv[index] = der
    if any(~index): deriv[~index] = np.nan
    return deriv

def flashprint(string):
    print(string, end='\r')
    sys.stdout.flush()

def get_scale(header):
    '''
    Funtion that given a header returns the pixel scale in arcsec/pixel.
    '''
    wcs = WCS(header)
    scale = np.abs(np.mean(utils.proj_plane_pixel_scales(wcs))*3600)
    return scale

# def sim_rdn_stars(shape, nstars, fwhm, I0):
#     '''
#     Simulates a image with nstars randomly distributed in the image
#     with same intensity I0 and a gaussian PSF with fwhm.
#     '''
#     image = np.zeros(shape)
#     xrdn = np.random.randint(0,shape[0],nstars)
#     yrdn = np.random.randint(0,shape[1],nstars)
#     image[xrdn,yrdn] = I0
#     psf = gaussian(fwhm, shape)
#     stars = convolve2d(image,psf,mode='same')
#     stars = I0 * (stars - np.nanmin(stars)) / np.nanmax(stars)
#     return stars


def find_center(data, center, width=30):
    '''
    Find the center of an object in a cropped image.
    It uses centroid_quadratic to find the center of the object.
    It fist a quadractic function to the data and then
    finds the maximum of the function.

    Parameters
    ----------
        data: array
            Image data.
        center: tuple
            Center of the object in the image.
        width: int, optional
            Width of the cropped image.
    
    Returns
    -------
        x,y: tuple
            Center of the object in the image.
    '''    

    crop = data[np.int32(center[1]-width):np.int32(center[1]+width),
                    np.int32(center[0]-width):np.int32(center[0]+width)]

    x,y = centroid_quadratic(crop)

    x += np.int32(center[0]-width)
    y += np.int32(center[1]-width)
    return x,y

def average_bin(x,y,bins, method='mean'):
    '''Average a variable y in bins of x in 1D.
    
    Parameters
    ----------
        x : array
            x-axis values.
        y : array
            y-axis values.
        bins : array
            Bins to use.
        method: string
            Method to use for average, mean or median.
    
    Returns
    -------
        ybins : array
            Averaged y values in bins of x.
        rmsBinned:   array
            RMS of y binned values.
        rmsMean:   float
            Average (from method) of all rms values 
    '''
    comb = np.nanmean if method=='mean' else np.nanmedian
    x = np.zeros_like(x) + np.array(x)
    y = np.zeros_like(y) + np.array(y)
    args = np.digitize(x,bins)
    ybins = np.array([comb(y[args==i]) for i in range(1,len(bins))])
    mean_r = np.array([ybins[i-1] if 0 < i < len(ybins) + 1 else np.nan for i in args])
    rms = np.sqrt((y-mean_r).astype(np.float32)**2)
    rmsBined = np.array([comb(rms[args==i+1]) for i in range(len(ybins))])
    rmsMean = comb(rms)
    return ybins, rmsBined, rmsMean


def limits(x,y,n=30):
    index = ~np.isnan(y)    
    p = np.poly1d(np.polyfit(x[index],y[index],n))
    mins = argrelextrema(p(x), np.greater)
    return np.nanmin(mins)

def adaptive_histogram(data, bandwidth=None, weights=None):
    """
    Compute histogram based on a top-hat kernel with the specified (adaptive) bandwidth.
    :param data: Collection of data points
    :param bandwidth: width of the top-hat kernel
    """
    if bandwidth is None:
        h1 = 1 + int(np.sqrt(data.size))
        h2 = 1 + int(data.size / 2)
        h = 1 + int(np.sqrt(h1 * h2))
    else:
        h = bandwidth
    if weights is None:
        weights = np.ones_like(data)

    sorted_data = np.argsort(data.flatten())
    x = data.flatten()[sorted_data]
    w = weights.flatten()[sorted_data]
    cumul_mass = np.cumsum(w)

    x_mid = (x[h:] + x[:-h]) / 2
    # x_mean = (x_cumul[h:] - x_cumul[:-h]) / h
    half_h = int(h/2)
    x_median = x[half_h:half_h-h]
    density = (cumul_mass[h:] - cumul_mass[:-h]) / (x[h:] - x[:-h])
    x_bin = np.sqrt(x_mid*x_median)
    return x_bin, density / data.size

def change_coordinates(positions, center, pa):
    rotMatrix = np.array([[np.cos(pa),-np.sin(pa)],
                        [np.sin(pa),np.cos(pa)]])
    newCoordinates = np.matmul(np.transpose(positions-center[:,np.newaxis]),rotMatrix)
    return np.transpose(newCoordinates)

from scipy import ndimage
from matplotlib.colors import LogNorm

# Funtion that given an numerical distribution finds the FWHM of the distribution
def getFWHM(x,y, oversamp=100, height=False):
    '''
    Given a numerical distribution finds the FWHM of the distribution
    interpolation is done to overcome undersampling offsets
    
    Parameters
    ----------
        x : numpy.ndarray
            x-axis of the distribution
        y : numpy.ndarray
            y-axis of the distribution
        oversamp : int
            oversampling factor
        height : bool
            if True returns the height of the distribution at the FWHM

    Returns
    -------
        fwhm : float
            FWHM of the distribution
    '''
    xx = np.linspace(x[0],x[-1],len(x)*oversamp)
    yy = np.interp(xx,x,y)

    max_y = np.max(yy)
    half_max = max_y/2
    idx = np.where(yy >= half_max)[0]
    fwhm = xx[idx[-1]] - xx[idx[0]]
    if np.abs(yy[idx[-1]] - yy[idx[0]]) > max_y/2*len(idx):
        raise Warning('FWHM is not well defined, check your distribution')
    fwhm_y = (yy[idx[-1]] + yy[idx[0]])/2
    
    if not height: return fwhm
    if height: return fwhm, fwhm_y

def merge_pdf(infnList, outfn):
    '''Given a list of PDF files, 
    merge them in one pdf. 
    
    PARAMETERS
    ----------
        infnlist: list
            list of pdf filenames
        outfn: str
            name of the merge file
    
    RETURNS
    -------
        boolean: 
            True if it was created
    '''
    merger = PdfMerger()
    for infn in infnList:
        merger.append(PdfReader(infn, 'rb'))
    merger.write(outfn)
    return os.path.isfile(outfn)

def find_mode(x, weights=None):
    """
    Locate the mode of a distribution by fitting a polynomial within +-one-sigma interval.
    :param x: Collection of data points
    """
    # Legendre polynomials
    def L0(x): return np.ones_like(x)
    def L1(x): return x
    def L2(x): return (3 * x ** 2 - 1) / 2
    # def L3(x): return (5*x**3 - 3*x) / 2
    # def L4(x): return (35*x**4 - 30*x**2 + 3) / 8
    def norm_L(n): return 2 / (2 * n + 1)

    if weights is None:
        weights = np.ones_like(x)

    total_mass = np.nansum(weights)
    x0 = np.nansum(x*weights) / total_mass
    sigma = np.sqrt(np.nansum((x-x0)**2 * weights)/total_mass)
    delta_peak = 0
    i = 0
    while True:
        i += 1
        x0 += delta_peak
        delta = x - x0
        w = weights[np.abs(delta) < sigma]
        delta = delta[np.abs(delta) < sigma] / sigma
        # scalar product:
        c0 = np.mean(L0(delta)) / norm_L(0) # point density · L0
        c1 = np.mean(L1(delta)) / norm_L(1)  # point density · L1
        c2 = np.mean(L2(delta)) / norm_L(2)  # point density · L2
        # c3 = np.mean(L3(delta)) / norm_L(3)  # point density · L3
        # c4 = np.mean(L4(delta)) / norm_L(4)  # point density · L4
        total_mass = np.nansum(w)
        c0 = np.nansum(w*L0(delta))/total_mass / norm_L(0) # point density · L0
        c1 = np.nansum(w*L1(delta))/total_mass / norm_L(1) # point density · L1
        c2 = np.nansum(w*L2(delta))/total_mass / norm_L(2) # point density · L2
        polynomial_fit = c0*L0(delta) + c1*L1(delta) + c2*L2(delta)  # + c3*L3(delta) + c4*L4(delta)
        delta_peak = delta[np.argmax(polynomial_fit)] * sigma/np.sqrt(i)  # divide by sqrt(i) to prevent oscillations
        # print(x0, delta_peak)
        if np.abs(delta_peak) <= sigma/x.size:
            return x0, x0+delta*sigma, polynomial_fit*delta.size/x.size/sigma


def absoluteMagnitude(distance, apparent_mag):
    '''Transform apparent magnitude to absolute magnitude
    As apparent mag - 5*np.log10(distance) + 5.
    
    Parameters
    ----------
        distance : float
            Distance in pc
        apparent_mag : float
            Apparent magnitude
    
    Returns
    -------
        absolute_mag : float'''
    return apparent_mag - 5*np.log10(distance) + 5


def ttype_iband_offset(ttype, rms=0.19):
    '''
    Function that converts i-band magnitude to 3.6mu m magnitude.
    Using Juan Manuel Falcón Ramirez's TFG results. 
    
    Parameters:
    ----------
        ttype : float
                T-type of the galaxy.
    
    Returns:
    -------
        offset : float
                Offset to be added to the i-band magnitude.
    '''

    offset = 0.0070*ttype - 0.009*ttype + 0.152
    e_offset =  0.0004*ttype -  0.004*ttype + 0.011
    e_offset = np.sqrt(e_offset**2 + rms**2)
    return offset

def mass_iband_offset(mass, rms=0.17):
    '''
    Function that converts i-band magnitude to 3.6mu m magnitude.
    Using Juan Manuel Falcón Ramirez's TFG results. 
    
    Parameters:
    ----------
        mass : float
                Log Mass of the galaxy in Solar Masses.
    
    Returns:
    -------
        offset : float
                Offset to be added to the i-band magnitude.
        e_offset: float
                Error on the offset given the rms of the relation.
    '''

    offset = -0.052*mass*mass + 0.67*mass - 1.20
    e_offset = -0.007*mass*mass + 0.13*mass - 0.61
    e_offset = np.sqrt(e_offset**2 + rms**2)
    return offset, e_offset

def magnitude_to_mass(magnitude, distance, mass_to_light=0.6, absmagsolar = 6.02):
    '''
    Function that converts apparent magnitude to mass in solar masses.
    Parameters:
    ----------
        magnitude : float
                Apparent magnitude of the galaxy.
        distance : float
                Distance to the galaxy in Mpc.
        mass_to_ratio : float
                Mass to luminosity ratio. Default is 0.6.
        absmagsolar : float
                Absolute magnitude of the sun. Default is 6.02. Mag at 3.6 micron.
                Reference: Willmer 2018. DOI 10.3847/1538-4365/aabfdf
    Returns:
    -------
        mass : float
                Log Mass of the galaxy in solar masses.
    '''
    absmag = absoluteMagnitude(distance*1e6, magnitude)    
    luminosity =  10**(-0.4*(absmag - absmagsolar))
    mass = mass_to_light*luminosity
    return  np.log10(mass)




def optical_to_IR(mag, distance, epsilon=1e-5, mass_to_light=0.6, N=100, verbose=False):
    '''
    Iterative process to find the offset between optical i-band and 3.6 micron magnitudes
    using the relation between mass and color. 

    Parameters:
    ----------
        mag : float
                Magnitude of the galaxy in i-band.
        distance : float
                Distance to the galaxy in Mpc.
        epsilon : float
                Precision of the iterative process. Default is 1e-5.
        mass_to_light : float
                Mass to luminosity ratio. Default is 0.6.
        N : int
                Maximum number of iterations. Default is 100.
        verbose : bool
                If True, prints the offset and mass at each iteration at index=0.
    Returns:
    -------
        offset : float
                Offset to be added to the i-band magnitude.
    '''
    n = len(mag) if mag is np.ndarray else 1
    masses, offsets = np.zeros((2,N,n))
    masses[0,:] =  magnitude_to_mass(mag, distance, mass_to_light=mass_to_light)
    epsilon , difference, i = 1e-5, np.ones_like(mag), 0
    while (i < N-1) and (difference > epsilon).any():
        offsets[i+1,:],e_offset = mass_iband_offset(masses[i,:])
        masses[i+1,:] = magnitude_to_mass(mag + offsets[i+1,:], distance, mass_to_light=mass_to_light)
        difference = np.abs(offsets[i,:] - offsets[i+1,:])
        if verbose:
            print(f'{i:2d}  {offsets[i,0]:.5f}   {masses[i,0]:.5f}  {difference[0]:.5f}') 
        i+=1
    if i==N-1:
        warnings.warn('Maximum number of iterations reached. Convergence not reached.')
    return offsets[i,:], e_offset


def find(inDIR='',filetype='*.fits'):

    fileList = []
    
    # Walk through directory
    for dName, sdName, fList in os.walk(inDIR):
        for fileName in fList:
            if fnmatch.fnmatch(fileName, filetype): # Match search string
                fileList.append(os.path.join(dName, fileName))

    return fileList

# Computes the power spectrum of an image returns the power spectrum and the corresponding frequencies
# def power_spectrum(image):
#     # Compute the 2D power spectrum
#     f = np.fft.fft2(image)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     # Compute the 1D power spectrum
#     psd2D = np.abs(fshift)**2
#     psd1D = np.mean(psd2D, axis=0)
#     # Compute the corresponding frequencies
#     Lx = image.shape[0]
#     Ly = image.shape[1]
#     fx = np.fft.fftfreq(Lx, d=1)
#     fy = np.fft.fftfreq(Ly, d=1)
#     fX, fY = np.meshgrid(fx, fy)
#     f = np.sqrt(fX**2 + fY**2)
#     ind = np.argsort(f, axis=None)
#     f_sorted = f.flatten()[ind]
#     psd1D_sorted = psd1D.flatten()[ind]
#     return psd1D_sorted, f_sorted


def power_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f) 
    power = np.abs(fshift)
    k_y = np.fft.fftshift(np.fft.fftfreq(fshift.shape[0], d=2))
    k_x = np.fft.fftshift(np.fft.fftfreq(fshift.shape[1], d=2))
    k2 = (k_y**2)[:,np.newaxis] + (k_x**2)[np.newaxis,:]
    return power.flatten(), k2.flatten()



