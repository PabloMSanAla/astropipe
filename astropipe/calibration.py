import glob

import numpy as np
import os
from os.path import join
from astropy.stats import SigmaClip
import astroalign
import subprocess

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from .classes import AstroGNU
from .utils import get_pixel_scale

from tqdm import tqdm


class astrometry():
    '''
    Class to run astrometry from Python
    '''
    def __init__(self, telescope='TSS', sex=None, solve_field=None, config=None):
        
        self.config = config
        self.solve_field = solve_field
        self.sex = sex

        if not self.config:
            self.config = '/usr/local/Cellar/astrometry-net/0.93/etc/astrometry.cfg'
        if not self.solve_field:
            self.solve_field = '/usr/local/bin/solve-field'
        if not self.sex:
            self.sex = '/usr/local/bin/sex'
        
        if telescope: self.define_telescope(telescope)
        
    def __call__(self, filename, out=None, hdu=0, ra=None, dec=None, verbose=False):
        ''' Run astrometry on a filename. 
        
        Parameters:
        ----------
            filename: str
                Path to the file to be solved.
            out: str
                Path to the output file. If None, the output will add _astrom.
            hdu: int
                HDU to be solved.
            ra: float
                RA of the center of the image.
            dec: float
                DEC of the center of the image.
            verbose: bool
                If True, print the command to be executed.

        Returns:
        -------
            int: 0 if the command was executed successfully.  
        '''
        if ra: self.ra = ra
        if dec: self.dec = dec
        self.out = out
        if not self.out:
            file, ext = os.path.splitext(filename)
            self.out = file + '_astrom' + ext

        if ra and dec:
            self.cmd = f'''
            {self.solve_field} {filename} --no-plots -L {self.L} -H {self.H} -u arcsecperpix --overwrite \
            --source-extractor-path {self.sex} \
            --extension {hdu} --config {self.config} --resort --ra {self.ra} \
            --dec {self.dec} --radius {self.radius} -Unone --temp-axy -Snone -Mnone -Rnone -Bnone\
            -N {self.out}'''
        else:
            self.cmd = f'''
            {self.solve_field} {filename} --no-plots -L {self.L} -H {self.H} -u arcsecperpix --overwrite \
            --source-extractor-path {self.sex} \
            --extension {hdu} --config {self.config} 
            -N{self.out}'''
        

        if verbose: print(self.cmd)
        return os.system(self.cmd)


            
    def define_telescope(self,telescope):
        if telescope=='TSS':
            self.L = 0.9            # lower in arcsec/pixel
            self.H = 1.1
            self.radius = 0.5       # radius in degrees
        
        elif telescope=='INT':
            self.L = 0.3
            self.H = 0.4
            self.radius = 0.4
        elif telescope=='NTT':
            self.L = 0.2
            self.H = 0.26
            self.radius = 0.05

        else:
            print('Telescope not defined.')

# Make an structure to find out if there are darks
# or bias to make False or True
class structure():

    def __init__(self, night, band=None, extension='.fits'):
        self.night = night
        self.directories = os.listdir(night)
        self.extension = extension
        self.check_dark()
        self.check_bias()
        self.check_light(band=band)
        self.check_domeflat()
        self.check_skyflat()
        self.make_calibrated()

    def check_dark(self):
        bool_array = ['dark' in f.lower() for f in self.directories]
        if any(bool_array):
            self.dark = join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.darkList = glob.glob(join(self.dark,'*'+self.extension))
        else:
            self.dark = None
    
    def check_bias(self):
        bool_array = ['bias' in f.lower() for f in self.directories]
        if any(bool_array):
            self.bias = join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.biasList = glob.glob(join(self.bias,'*'+self.extension))
        else:
            self.bias = None
        
    def check_light(self, band=None):
        bool_array = [f.lower().startswith(('li','sc')) for f in self.directories]
        if any(bool_array):
            self.light = join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            if band: 
                dirs = [s for s in os.listdir(self.light)]
                anyband =[band.lower() in f.lower() for f in dirs]
                if any(anyband):
                    self.light = join(self.light,dirs[np.argwhere(anyband)[0][0]])
            self.lightList = glob.glob(join(self.light,'*'+self.extension))
        else:
            self.light = None
    
    def check_domeflat(self):
        bool_array = ['dome' in f.lower() for f in self.directories]
        if any(bool_array):
            self.domeflat = join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.domeflatList = glob.glob(join(self.domeflat,'*'+self.extension))
        else:
            self.domeflats = None

    def check_skyflat(self):
        bool_array = ['sky' in f.lower() for f in self.directories]
        if any(bool_array):
            self.skyflat = join(self.night,
            self.directories[np.argwhere(bool_array)[0][0]])
            self.skyflatList = glob.glob(join(self.skyflat,'*'+self.extension))
        else:
            self.skyflats = None

    def make_calibrated(self):
        self.calibrated = join(self.night,'calibrated')
        if not os.path.exists(self.calibrated):
            os.mkdir(self.calibrated)
    
    def set_masterdomeflat(self,file):
        self.masterbias = file
    
    def set_masterskyflat(self,file):
        self.masterskyflat = file
    
    def set_masterbias(self,file):
        self.masterbias = file

    def set_masterdark(self,file):
        self.masterbias = file

def deg_to_hms(ra_deg,dec_deg):
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
    return coord.to_string(style="hmsdms", precision=2, pad=True)

def deg_to_dms(dec_deg):
    coord = SkyCoord(ra=0, dec=dec_deg, unit="deg")
    return coord.to_string(style="hmsdms", precision=2, pad=True)

from astropy.wcs import WCS

def get_corners(fileList):
    '''
    Get the equatorial coordinates (ra,dec) of the corners of 
    the footprint of a list of images. Also returns the mean pixel scale.

    Parameters
    ----------
        fileList : list
            List of images.
    
    Returns
    -------
        min_ra, max_ra, min_dec, max_dec, scale : float
            Minimum and maximum values of RA and Dec [deg] and 
            mean pixel scale [pixel/arcsec].
    '''
    ras, decs, pixscales = [], [], []
    for file in fileList:
            header = fits.getheader(file)
            wcs = WCS(header)
            height, width = fits.getdata(file).shape
            pixels_corners = np.array([[0, 0], [0, height], [width, height], [width, 0]])
            ra_dec_corners = wcs.pixel_to_world_values(pixels_corners[:, 0], pixels_corners[:, 1])
            ras.extend(ra_dec_corners[0])
            decs.extend(ra_dec_corners[1])
            pixscales.append(get_pixel_scale(header))

    min_ra, max_ra = np.nanmin(ras), np.nanmax(ras)
    min_dec, max_dec = np.nanmin(decs), np.nanmax(decs)

    return min_ra, max_ra, min_dec, max_dec, np.nanmean(pixscales)

def change_config(file, params, length=23):
    with open(file,'r') as f:
        lines = f.readlines()
    for key,value in params.items():
        for i,line in enumerate(lines):
            if line.startswith(key+' '):
                lines[i] = f'{key.ljust(length)} {value}\n'
    with open(file,'w') as f:
        f.writelines(lines)


def stack(image_list, hdu=0, sigma=3, maxiters=3, dtype=np.float32):
    '''
    Given a file list of images stack them using sigma clipping. 

    '''
    shape = fits.getdata(image_list[0],hdu).shape + (len(image_list),)
    stack = np.zeros(shape, dtype=dtype)
    for i in range(shape[2]):
        stack[:,:,i] = fits.getdata(image_list[i],hdu)
    sigmaclip = SigmaClip(sigma=sigma,maxiters=maxiters)
    master = np.ma.mean(sigmaclip(stack, axis=2),axis=2)
    master[np.where(master.mask)] = np.nan
    return master.data


def darkstack(dark_list, masterbias = 0, hdu=0, sigma=3, maxiters=3):
    masterdark = np.dstack((fits.getdata(dark_list[0],hdu)-masterbias,
                      fits.getdata(dark_list[1],hdu)-masterbias))
    for image in dark_list[2:]:
        masterdark = np.dstack((masterdark,fits.getdata(image,hdu) - masterbias))
    sigmaclip = SigmaClip(sigma=sigma,maxiters=maxiters)
    masterdark = np.ma.mean(sigmaclip(masterdark, axis=2),axis=2)
    masterdark[np.where(masterdark.mask)] = np.nan
    return masterdark.data

def flatstack(flat_list, masterbias=0, masterdark=0, hdu=0, dtype=np.float32):
    sc_norm = SigmaClip(sigma=2,maxiters=3)
    sc_comb = SigmaClip(sigma=3,maxiters=3)
    shape = fits.getdata(flat_list[0],hdu).shape + (len(flat_list),)
    masterflat = np.zeros(shape,dtype=dtype)
    for i in range(shape[2]):
        masterflat[:,:,i] = fits.getdata(flat_list[i],hdu) - masterbias - masterdark
        norm = np.ma.mean(sc_norm(masterflat[:,:,i]))
        masterflat[:,:,i] /= norm
        
    masterflat = np.ma.mean(sc_comb(masterflat,axis=2),axis=2)
    masterflat[np.where(masterflat.mask)] = np.nan
    return masterflat.data

def calibrate(lights_list, masterdark=0, masterbias=0, masterflat=1, 
                           hdu=0, keytime='EXPTIME', dir=None, mask=True, dtype=np.float32):
    calibrated_list = []
    iterpath = True if dir is None else False

    for light in lights_list:
        try:
            if iterpath: dir = os.path.dirname(light)
            header = fits.getheader(light,hdu)
            time = float(header[keytime]) if keytime in header else 1
            calibrate = (fits.getdata(light,hdu)-masterdark-masterbias)
            calibrate = correct_flat(calibrate, masterflat, mask=mask) / time
            header = fits.getheader(light, hdu)
            calibrated_list.append(join(dir,os.path.basename(
                        light).replace('.fits','_calibrated.fits')))
            save_fits(calibrate.astype(dtype),header,
                    calibrated_list[-1])
        except Exception as e: 
            print('Error in calibration of',light, ':')
            print(e)
        
    return calibrated_list

def calibrate_night(night, masterdark=None, masterbias=None):
    if night.bias:
        masterbias = stack(glob.glob(join(night.bias,'*.fit*')))
        save_fits(masterbias,None,join(night.bias,'masterbias.fits'))
    else:
        masterbias = 0
    if night.dark:
        masterdark = darkstack(glob.glob(join(night.dark,'*.fit*')),
                                masterbias)
        save_fits(masterdark,None,join(night.dark,'masterdark.fits'))
    else:
        masterdark = 0
    
    for light in glob.glob(join(night.light,'*.fit*')):
        calibrate = fits.getdata(light,0)-masterdark-masterbias
        header = fits.getheader(light,0)
        save_fits(calibrate,header,
                join(night.calibrated,os.path.basename(light).split('.')[-2]+'_calibrated.fits'))

def save_fits(data,header,path, dtype=np.float32):
    hdu = fits.PrimaryHDU(data.astype(dtype), header)
    hdu.writeto(path,overwrite=True)
    
def correct_flat(data,flat,mask=True):
    corrected = data/flat 
    if mask:
        corrected[np.where(corrected>np.nanmax(data))] = np.nan
        corrected[np.where(corrected<np.nanpercentile(data[np.where(data<0)],10))] = np.nan
    return corrected

def noise_parrallel(filelist, config_nc='', ncpu=8, out=''):
    text = ''
    for i in filelist:
        text = text + f'" astnoisechisel {i} {config_nc} -o{join(out,os.path.basename(i).replace(".fits","_detected.fits"))}"\n '
    
    cmd = f'''
#! /bin/bash

commands=(
{text}
)

parallel --jobs {ncpu}'''+ ' ::: "${commands[@]}"'

    return cmd



def autoflat_parallel(flat_files, masterflat=None, config='', hdu=0, ncpu=8, divider=1):
    """Creates an autoflat given a list of science images. 
    We first run noisechisel in parallel to mask sources in the images.
    Then we combine the images in chunks to reduce memory usage.
    """
    
    # Create bash file to run noisechisel in paraller
    out = join(os.path.dirname(flat_files[0]),'noisechisel')
    cmd = noise_parrallel(flat_files, config_nc=config, N=ncpu, out=out)
    noisefile = join(os.path.dirname(flat_files[0]),'noisechisel.sh')
    with open(noisefile,'w') as f:
        f.write(cmd)
    cmd = ['bash', noisefile]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Start building the flat in chunks to keep memory usage low
    
    sc_norm = SigmaClip(sigma=2,maxiters=3)
    sc_comb = SigmaClip(sigma=3,maxiters=3)

    if masterflat is None:
        reference = flat_files[np.random.randint(0,len(flat_files),2)[0]]
        reference_data = fits.getdata(reference,hdu)
        detections = fits.getdata(join(out,os.path.basename(reference).replace('.fits','_detected.fits')),1)
        reference_data =  (1-detections) * reference_data
        reference_data[np.where(reference_data==0)] = np.nan
        reference_data /= np.ma.mean(sc_norm(reference_data))
        masterflat = 1
    else:
        reference_data = masterflat

    shape = reference_data.shape
    flat = np.zeros(shape)
    xdiv = np.linspace(0,reference_data.shape[1],divider+1).astype(np.int32)

    for i in range(1, divider+1):
        tempshape = reference_data[:,xdiv[i-1]:xdiv[i]].shape + (len(flat_files),)
        regionflat = np.zeros(tempshape) 
        k=0
        for file in flat_files:
            try:
                regionflat[:,:,k] = fits.getdata(file,hdu=hdu)[:,xdiv[i-1]:xdiv[i]] * (
                            1-fits.getdata(join(out,os.path.basename(file).replace('.fits','_detected.fits')),2)[:,xdiv[i-1]:xdiv[i]])
                regionflat[:,:,k][np.where(regionflat[:,:,i]==0)]=np.nan
                norm = np.ma.mean(sc_norm(regionflat[:,:,i] / reference_data[:,xdiv[i-1]:xdiv[i]]))
                regionflat[:,:,k] = regionflat[:,:,i]/norm
            except:
                print(f'Error in {file}')
            k+=1
        regionflat = np.ma.mean(sc_comb(regionflat, axis=2),axis=2)
        regionflat[np.where(regionflat.mask)]=np.nan
        flat[:,xdiv[i-1]:xdiv[i]] = regionflat.data
    
    return flat
    

def autoflat(flat_files, mask_files, masterflat=None, hdu=0, dtype=np.float32):

    sc_norm = SigmaClip(sigma=2,maxiters=3)
    sc_comb = SigmaClip(sigma=3,maxiters=3)
    
    if masterflat is None:
        random_index = np.random.randint(0,len(flat_files),1)[0]
        reference = flat_files[random_index]
        reference_data = fits.getdata(reference,hdu)
        mask = fits.getdata(mask_files[random_index])
        reference_data =  (1-mask) * reference_data
        reference_data[np.where(reference_data==0)] = np.nan
        reference_data /= np.ma.mean(sc_norm(reference_data))

        masterflat = 1
        
    else:
        reference_data = masterflat

    flat = np.zeros(reference_data.shape + (len(flat_files),), dtype=dtype)
    
    i=0
    for file,maskfile in tqdm(zip(flat_files, mask_files), total=len(flat_files), desc='Building flat...'):
        try:
            data = fits.getdata(file,hdu=hdu) 
            mask = fits.getdata(maskfile)
            flat[:,:,i] = (1-mask) * data
            flat[:,:,i][np.where(flat[:,:,i]==0)]=np.nan
            norm = np.ma.mean(sc_norm(flat[:,:,i] / reference_data))
            flat[:,:,i] = flat[:,:,i]/norm
        except Exception as e:
            print(f'Error in {file}: {e}')
        i+=1

    masterflat = np.ma.mean(sc_comb(flat, axis=2),axis=2)
    masterflat[np.where(masterflat.mask)]=np.nan

    return masterflat.data
        


def register(list, nref=0,hdu=0,verbose=False):
    """
    Function to register a list of files.
    list: list of files to be registered

    return:
    list_registered: list of images registered
    """
    if verbose: print(f'Start aligment for {len(list)} images')
    if verbose: print(f'Use {list[nref]} for reference image...')
    reference = fits.getdata(list[nref],hdu)
    align_list = [reference]
    del(list[nref])
    for file in list:
        if verbose: print(f'Aligning {file}...')
        target = fits.getdata(file,hdu)
        p, (pos_img, pos_img_rot) = astroalign.find_transform(target, reference)
        aligned = astroalign.register(target, reference)
        align_list.append(aligned)
    if verbose: print(f'Alignment finished.')
    return align_list

def register_arrays(list, nref=0,verbose=False):
    """
    Function to register a list of images.
    list: list of images to be registered

    return:
    list_registered: list of images registered
    """
    if verbose: print(f'Start aligment for {len(list)} images')
    if verbose: print(f'Use {list[nref]} for reference image...')
    reference = list[nref]
    align_list = [reference]
    for target in list:
        p, (pos_img, pos_img_rot) = astroalign.find_transform(target, reference)
        aligned = astroalign.register(target, reference)
        align_list.append(aligned)
    if verbose: print(f'Alignment finished.')
    return align_list
