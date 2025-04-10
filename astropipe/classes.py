
from astropy.io import fits
from astropy.wcs import WCS,utils
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.table import Table

from datetime import datetime
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt
import platform
import os
from os.path import join

from lmfit.models import GaussianModel

import copy

from ctypes import c_float, c_double
from mtolib import _ctype_classes as ct
from mtolib.tree_filtering import filter_tree, get_c_significant_nodes, init_double_filtering
from mtolib import postprocessing
import mtolib.main as mto

from .plotting import noise_hist, make_cmap, show
from .utils import *
from .profile import background_estimation_euclid, isophotal_photometry, elliptical_radial_profile

import astropipe
path_to_package = os.path.dirname(astropipe.__file__)


import sys
import sewpy


class Image:
    '''
    Image class to work with astronomical images. 

    Attributes
    ----------
        data :
    '''
    def __init__(self, filename, hdu=0, zp=22.5):
        '''
        Initialize class by reading image fits file. 
        
        Parameters
        ----------
        filename : str
            Name of the image fits file
        hdu : int, optional
            HDU exten
        '''

        self.data = fits.getdata(filename, hdu)
        self.header = fits.getheader(filename, hdu)
        self.wcs = WCS(self.header)
        self.pixel_scale = np.mean(utils.proj_plane_pixel_scales(self.wcs)*3600)
        self.file = filename
        self.directory = os.path.dirname(self.file)
        self.name, self.extension = os.path.splitext(self.file)
        self.name = os.path.basename(self.name)
        self.bkg, self.bkgstd = 0,0
        self.zp = zp 
        self.hdu = hdu

    def obj(self, ra, dec):
        '''Defines de equatorial coordinates of the object
        of interest in the image.
        Parameters
        ----------
            ra : float
                Right ascension of the object in degrees.
            dec : float
                Declination of the object in degrees.
        Returns
        -------
            None
        '''
        self.ra = ra
        self.dec = dec
        self.SkyCoord = SkyCoord(ra, dec, frame="fk5", unit="deg")
        self.pix = skycoord_to_pixel(
            self.SkyCoord, self.wcs
        )
        self.pix = np.array([np.float64(self.pix[0]),
                            np.float64(self.pix[1])])
        self.x,self.y = np.int64(self.pix)

    def obj_id(self, mask):
        '''
        Finds object ID in a given segmentation mask
        an saves it in an attribute.

        Parameters
        ----------
            mask : numpy array
                Segmentation mask of the image.
        Returns
        -------
            None

        '''
        self.id = mask.objects[int(self.pix[1]), int(self.pix[0])]

    def sky_to_pixel(self, ra, dec):
        '''
        Use the WCS of the image to convert sky coordinates
        to pixel coordinates. Using astropy SkyCoord.

        Parameters
        ----------
            ra : float
                Right ascension of the object in degrees.
            dec : float
                Declination of the object in degrees.
        
        Returns
        -------
            (xp, yp) : numpy.ndarray
                Pixel coordinates of the object.
        '''
        return skycoord_to_pixel(SkyCoord(ra, dec, frame="fk5", unit="deg"), self.wcs)

    def pixel_to_sky(self, xp, yp):
        '''
        Convert pixel coordinates to sky coordinates using
        the WCS of the image. Using astropy pixel_to_skycoord.

        Parameters
        ----------
            xp : float
                x coordinate of the object in pixels.
            yp : float
                y coordinate of the object in pixels.
        
        Returns
        -------
            (ra, dec) : ~astropy.coordinates.SkyCoord
                Sky coordinates of the object.
        '''
        return pixel_to_skycoord(xp, yp, self.wcs)

    def noise(self,mask,plot=False):
        noise = self.data[where([mask==0, self.data != self.data[0,0]])]
        mu = np.nanmean(noise)
        std = np.nanstd(noise)
        hist = np.histogram(noise,bins=500,range=[mu-6*std,mu+6*std])
        model = GaussianModel()
        params = model.guess(hist[0],x=hist[1][1:],)
        result = model.fit(hist[0],params,x=hist[1][1:]) 
        self.std = result.values['sigma']
        self.bkg = result.values['center']
        self.header['STD'] = self.std
        self.header['MAG_LIM'] = mag_limit(std, Zp=self.zp, scale=self.pixel_scale)
        if plot:
            noise_hist(result,out=plot)
        
    def crop(self, center, width=(500,500)):
        '''
        Use the astropipe.utils.crop function to crop the image
        given a center and width. It updates the data and header 
        attributes of the class preserving the WCS information 
        It also saves the parameters of the cropping procedoure 
        in an attribute.

        Parameters
        ----------
            center : tuple
                (x,y) coordinates of the center of the crop.
            width : tuple, optional
                (width_x, width_y) of the crop.
        
        Returns
        -------
            None
        '''

        self.data, self.header, cropRange = crop(self.data, self.header, center, width)
        
        self.cropParams = {'center': np.int64(center),
                            'width': np.int64(width),
                            'range': cropRange}
        self.wcs = WCS(self.header)

        if hasattr(self,'pix'):
            x,y = cropRange
            self.x = np.int64(center[0]-x[0])
            self.y = np.int64(center[1]-y[0])
            self.pix = np.array([self.x,self.y])
        
    def copy(self):
        '''Method to copy the class in another variable
        '''
        return copy.deepcopy(self)

    def radial_photometry(self,  growth_rate = 1.03, max_r = None, plot=None, save=None):
        '''Method to calculate the radial profile of the object
        using the morphological parameters of the object.
        
        Parameters
        ----------
            growth_rate : float, optional
                Growth rate of the radial bins.
            max_r : float, optional
                Maximum radius to calculate the profile.
            plot : str, optional
                Name of the file to save the plot.
            save : str, optional
                Name of the file to save the profile.
        
        Returns
        -------
            profile : astropipe.profile.Profile
                Radial profile of the object.'''

        max_r = 2*self.bkgrad if max_r is None else max_r
    
        profile = elliptical_radial_profile(self.data, max_r, self.pix, self.pa, self.eps,
                                         growth_rate=growth_rate, plot=plot, save=save)
        
        profile.set_params(bkg=self.bkg, bkgstd=self.bkgstd, 
                           zp=self.zp, pixscale=self.pixel_scale)
        profile.brightness()
        return profile

    def isophotal_photometry(self, max_r=None, plot=None, save=None, growth_rate=1.03,
                            fix_center=True, fix_pa=False, fix_eps=False):
        '''Method to calculate the radial profile of the object
        fitting the morphological parameters of the object for 
        each isophote.
        
        Parameters
        ----------
            growth_rate : float, optional
                Growth rate of the radial bins.
            max_r : float, optional
                Maximum radius to calculate the profile.
            plot : str, optional
                Name of the file to save the plot.
            save : str, optional
                Name of the file to save the profile.
            fix_center : bool, optional [True]
                Fix the center of the object to the center of the image.
            fix_pa : bool, optional [False]
                Fix the position angle of the object to the value in the header.
            fix_eps : bool, optional [False]
                Fix the ellipticity of the object to the value in the header.
        
        Returns
        -------
            profile : astropipe.profile.Profile
                Radial profile of the object.'''
        
        profile = isophotal_photometry(self.data, self.pix, self.pa, self.eps, self.reff,
                                    max_r=max_r, plot=plot, save=save, growth_rate=growth_rate,
                                    fix_center=fix_center, fix_pa=fix_pa, fix_eps=fix_eps)
        
        profile.set_params(bkg=self.bkg, bkgstd=self.bkgstd, 
                           zp=self.zp, pixscale=self.pixel_scale)
        profile.brightness()
        return profile
    
    def show(self, ax=None, vmin=None, vmax=None, cmap='nipy_spectral',
                    width=400, plotmask=True):
        '''Shows the surface brightness map of the image centered in the
        object of interest. 
        '''
        if not hasattr(self.data,'mask'): plotmask = False
      
        ax = show(self.data-self.bkg, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, plotmask=plotmask,
                  zp=self.zp, pixel_scale=self.pixel_scale)
        if hasattr(self,'x'):
            ax.set_xlim([self.x-width,self.x+width])
            ax.set_ylim([self.y-width,self.y+width])
        ax.text(0.02, 1, self.name.replace('_','\_'), horizontalalignment='left',
                verticalalignment='bottom', transform=ax.transAxes, fontweight='bold',fontsize='large')
        plt.tight_layout()
        return ax
    
    def get_background(self, growth_rate=1.05, init=1, out=None):
        '''
        Calculates the local background value around object using method
        implemented in astropipe.profile.background_estimation
        '''
        results = background_estimation_euclid(self.data, self.pix, self.pa, self.eps, init=init,
                                                                        plot=out, growth_rate=growth_rate)
        self.bkg, self.bkgstd, self.bkgrad  = results['ellip_bkg'], results['rect_bkgstd'], results['bkgrad']
        
    def get_morphology(self, nsigma=1):
        '''Calculates the morphological parameters of the object
        using a binarize image up to nsigma times the background. '''

        binary = binarize(self.data, nsigma=nsigma, center=(self.x,self.y))
        self.pa,self.reff,self.eps = morphology(binary)
        radcent = np.int64(self.reff/4) if  np.int64(self.reff/4)>20 and np.int64(self.reff/4)<80 else 20
        self.pix = find_center(self.data, self.pix, radcent)
        self.x,self.y = np.int64(self.pix)

    
    def set_mask(self,mask):
        self.data = ma.masked_array(ma.getdata(self.data), mask=mask)
    
    def set_data(self, data):
        if hasattr(self,'mask'):
            self.data = ma.masked_array(data, 
                mask=np.ma.getmask(self.data))
        else:
            self.data = data

    def set_morphology(self, pa=None, eps=None, reff=None, x=None,y=None):
        if pa: self.pa = pa
        if eps: self.eps = eps
        if reff: self.reff = reff
        if x: self.x = np.int64(x)
        if y: self.y = np.int64(y)
        if x and y: self.pix = np.array([self.x, self.y])

    def set_background(self, bkg=None, bkgstd=None, bkgrad=None):
        if bkg: self.bkg = bkg
        if bkgstd: self.bkgstd = bkgstd
        if bkgrad: self.bkgrad = bkgrad

    def set_catalog(self, table):
        self.catalog = table
    
    def set_std(self, std):
        self.std = std
        self.maglim = self.counts_to_mu(self.std)

    def set_maglim(self, mag_lim):
        self.maglim = mag_lim
        self.std = self.mu_to_counts(self.maglim)
    
    def set_extinction(self,Av):
        self.Av = Av
    
    def counts_to_mu(self,counts):
        return self.zp -2.5*np.log10(counts/self.pixel_scale**2)
    
    def mu_to_counts(self, mu):
        return 10**((self.zp-mu)/2.5)*self.pixel_scale**2


class SExtractor:
    
    '''
    Class to run SExtractor on a FITS image.
    
    To run SExtractor: 
        (1st) You need to have it installed on your computer
    
    This class lets you create an instance of SExtractor with 
    the parameters you set.

    Then, you run the class on a FITS image and it returns the 
    segmentation map and catalog from SExtractor.
    '''

    def __init__(self, params=None, config=None, sexpath=None):

        sex_param_path = os.path.abspath(join(path_to_package, '..','external','SExtractor'))
        self.files_default = {
            "FILTER_NAME": join(sex_param_path,'default.conv'),
            "PSF_NAME": join(sex_param_path, "default.psf"),
            "STARNNW_NAME": join(sex_param_path,"default.nnw"),
            'PHOT_FLUXFRAC': 0.9,
        }

        self.params_default = [
            "NUMBER",
            'X_IMAGE',
            'Y_IMAGE',
            "ALPHA_J2000",
            "DELTA_J2000",
            "MAG_ISO",
            "MAGERR_ISO",
            'MU_MAX',
            "BACKGROUND",
            "A_WORLD",
            "B_WORLD",
            "THETA_J2000",
            "CLASS_STAR",
            'FLUX_RADIUS',
            'KRON_RADIUS'
        ]

        self.params = self.params_default
        self.config = self.files_default
        self.method = 'SExtractor'

        if params is not None:
            self.add_params(params)
        if config is not None:
            self.add_config(config)

        if sexpath is None: 
            system = platform.system()
            if system=='Linux':
                sexpath = 'source-extractor'
            else:
                sexpath = 'sex'
        self.sexpath = sexpath

    def add_params(self, p_list):
        
        self.params = np.unique(self.params + p_list).tolist()

    def add_config(self, c_dict):
        for key in c_dict:
            self.config[key] = c_dict[key]

    def run(self, file, keep=False):
        sew = sewpy.SEW(params=self.params, config=self.config, sexpath=self.sexpath)
        self.wordir = sew.workdir
        if isinstance(file, str):
            self.file = file
        else:
            self.file = join(self.wordir,'sextractor_image.fits')
            fits.PrimaryHDU(file).writeto(self.file,overwrite=True)

        if 'CHECKIMAGE_NAME' not in self.config: self.config['CHECKIMAGE_NAME'] = join(self.wordir,'sex.fits')
        self.out = sew(self.file)
        self.catalog = self.out['table']
        self.seg_file = self.config['CHECKIMAGE_NAME']
        self.objects = fits.getdata(self.seg_file)
        if not keep: os.system(f'rm -r {self.wordir}')


    def cmap(self, background_color='#000000ff', seed=None):
        return make_cmap(np.nanmax(self.objects), background_color=background_color, seed=seed)
    

    def load_catalog(self, file):
        self.cat = Table.read(file, format='ascii.sextractor')

class MTObjects():
    '''API of the MTObjects program by Caroline Haigh.
    https://github.com/CarolineHaigh/mtobjects
    '''
    def __init__(self):

        self.out = 'out.fits'
        self.par_out='parameters.csv'
        self.soft_bias = 0
        self.gain = -1
        self.bg_mean = None
        self.bg_variance= -1
        self.alpha = 1e-6
        self.move_factor = 0.5
        self.min_distance = 0.0 
        self.verbosity = 0
    
    def setup(self, data):
        # Set the pixel type based on the type in the image
        self.d_type = c_float
        if np.issubdtype(data.dtype, np.float64):
            self.d_type = c_double
            init_double_filtering(self)

        # Initialise CTypes classes
        try:
            ct.init_classes(self.d_type)
        except:
            pass
    
    def run(self, data, save=False):
        self.setup(data)
        # Pre-process the image
        processed_image = mto.preprocess_image(data, self, n=2)

        # Build a max tree
        mt = mto.build_max_tree(processed_image, self)

        # Filter the tree and find objects
        id_map, sig_ancs = mto.filter_tree(mt, processed_image, self)

        # Relabel objects for clearer visualisation
        id_map = mto.relabel_segments(id_map, shuffle_labels=False)

        self.objects = id_map.reshape(data.shape) + 1
        if save:
            # Generate output files
            mto.generate_image(data, id_map, self)
            mto.generate_parameters(data, id_map, sig_ancs, self)
        masked = np.ma.array(data, mask=np.isnan(data))
        catalog = np.array(postprocessing.get_image_parameters(masked, id_map.ravel(), sig_ancs, self)) 
        self.catalog = Table(rows= catalog[1:], names= catalog[0], dtype=[np.int64]+[np.float32]*(len(catalog[1])-1))

    def cmap(self, background_color='#000000ff', seed=None):
        return make_cmap(np.int64(np.nanmax(self.objects)), background_color=background_color, seed=seed)

    def help(self):
        print('''
        This is a python wrapper for the MTObjects program by 
        Caroline Haigh. It is a program to detect objects in 
        astronomical images.

        The original code can be found at: https://github.com/CarolineHaigh/mtobjects
        
        The differents parameters are:

        -help	      Show the help message and exit
        -out	      Location to save filtered image. Supports .fits and .png filenames
        -par_out	  Location to save calculated parameters. Saves in .csv format
        -soft_bias	  Constant bias to subtract from the image
        -gain	      Gain (estimated by default)
        -bg_mean	  Mean background (estimated by default)
        -bg_variance  Background variance (estimated by default)
        -alpha	      Significance level - for the original test, this must be 1e-6
        -move_factor  Higher values reduce the spread of large objects. Default = 0.5
        -min_distance Minimum brightness difference between objects. Default = 0.0
        -verbosity	  Verbosity level (0-2). Default = 0.
        
        When running, two attributes are generated:
            objects: numpy array with the segmentation map labelled (0=sky)
            catalog: astropy table with the parameters of the objects 
                     catalog columns : ['ID', 'X', 'Y', 'A', 'B', 'theta',  # 'kurtosis',
                           'total_flux', 'mu_max', 'mu_median', 'mu_mean',
                           'R_fwhm', 'R_e', 'R10', 'R90','area']
                    
        ''')



class AstroGNU():
    def __init__(self, data, hdu=0, dir='', loc=''):
        if isinstance(data, str):
            self.file = data
            self.temp = False
        else:
            self.file = join(dir,'_temp.fits')
            new_hdu = fits.PrimaryHDU(data)
            new_hdu.writeto(self.file,overwrite=True)
            hdu=0
            self.temp = True

        if not dir: dir = os.path.dirname(self.file)
        self.directory = dir
        self.hdu = hdu
        self.name, self.extension = os.path.splitext(self.file)
        self.name = os.path.basename(self.name)
        self.loc = loc
        self.method = 'AstroGNU'

    def noisechisel(self, config='', keep=False):
        
        self.nc_file  = join(self.directory,self.name+'_nc.fits')
        
        self.nc_config = config
        self.nc_cmd = f'astnoisechisel {self.file} -h{self.hdu} {self.nc_config} -o{self.nc_file} -q'
        self.nc_cmd = join(self.loc,self.nc_cmd)

        os.system(self.nc_cmd)
        
        self.detections = fits.getdata(self.nc_file,'DETECTIONS')
        self.background = np.nanmean(fits.getdata(self.nc_file,'SKY'))

        if not keep: os.remove(self.nc_file)
        if self.temp: os.remove(self.file)

    def segment(self, config='', clumps=False, keep=False):
        self.seg_file  =  join(
                            self.directory, 
                            self.name+'_seg.fits')

        self.seg_config = config
        self.seg_cmd = f'astsegment {self.nc_file} {self.seg_config} -o{self.seg_file} -q'
        self.seg_cmd = join(self.loc,self.seg_cmd)
        os.system(self.seg_cmd)

        self.objects = fits.getdata(self.seg_file,'OBJECTS')
        

        if clumps: self.clumps = fits.getdata(self.seg_file,'CLUMPS')
        if not keep: os.remove(self.seg_file)


    def make_catalog(self,config='',params='',fracmax=0.05,zp=22.5):
        self.cat_file = join(self.directory,self.name+'_nc.fits')
        self.mkc_config = config
        self.mkc_cmd = f'astmkcatalog -irdmGnABp --fwhm --fracmaxradius1 --fracmax={fracmax} '
        self.mkc_cmd += f' {self.mkc_config} --zeropoint={zp} {self.seg_file} -o{self.cat_file} -q'
        self.mkc_cmd = join(self.loc,self.mkc_cmd)
        
        os.system(self.mkc_cmd)
                
        self.catalog = Table.read(self.cat_file)

    def remove(self, nc=True,seg=True,cat=True):
        if nc:  os.remove(self.nc_file)
        if seg: os.remove(self.seg_file)
        if cat: os.remove(self.cat_file)
        if self.temp: os.remove(self.file)
        


class Directories():
    '''Class to help keep track where all the products of the pipeline 
     is being save. It generates automatic names for mask, and profiles
    '''
    def __init__(self, name, path=None, create=True):
        '''Once initialize it creates the structures of directories where 
         the products will be save.'''
        if not path: path = os.path.dirname(name)
        self.out = join(path,'astropipe_'+name)
        if not os.path.exists(self.out) and create:
             os.mkdir(self.out)
        self.temp = join(self.out,'temp_'+name)
        if not os.path.exists(self.temp) and create:
             os.mkdir(self.temp)
        self.mask = join(self.out, f'{name}_mask.fits')
        self.profile = join(self.out, f'{name}_profile.fits')

    def create(self):
        '''Create the out and temp directories'''
        if not os.path.exists(self.out): os.mkdir(self.out)
        if not os.path.exists(self.temp): os.mkdir(self.temp)

    def set_regions(self,path):
        self.regions = path
    def set_mask(self, file):
        self.mask = file
    def set_profile(self, file):
        self.profile = file
        
class log_class:
    """Create a log file for the outputs of each execution of the pipeline."""

    def __init__(self):
        """ Initialize log file"""

        self.name = "log_{}.txt".format(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

        f = open(self.name, "w+")
        f.write(
            75 * "=" + "\n"
            "astropipe Log file" + "\n" + 75 * "=" + "\n"
        )
        f.close()

        self.insert_log = ">> " + self.name

    def insert_line(self):
        """Command line to import output in log file"""

        f = open(self.name, "a+")
        f.write(75 * "-" + "\n")
        f.close()


source_extractor_keys = {
    'DEBLEND_NTHRESH': '--partition-threshold-count',
    'DEBLEND_MINCONT': '--partition-threshold-fraction',
    'DETECT_MINAREA':'--partition-minimum-area',
    'DETECT_THRESH':'--detection-threshold',
    'BACK_SIZE':'--background-cell-size',
    'BACK_FILTERSIZE': '--smoothing-box-size',
    'CHECKIMAGE_NAME':'--check-image-segmentation',
    'CATALOG_NAME':'--output-catalog-filename',
    'MAG_ZEROPOINT':'--magnitude-zero-point',
}