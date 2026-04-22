import os
import wget
import glob
import numpy as np

from time import sleep
from urllib.request import urlopen
import subprocess as sp
from datetime import datetime

from scipy.interpolate import griddata

from astropy.io import fits 
from astropy.table import Table, hstack
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia
from astropy import coordinates as coords

from astropy.wcs import WCS, utils
from astropy import units as u




def sdss_script_mosaic(url):
    '''
    Function that downloads the script from the SDSS  
    SAS mosaic server and returns it as a string.

    Parameters
    ----------
        url : str
            URL of the script to download.

    Returns
    -------
        str: str
            String with the script.
    '''
    f = urlopen(url)
    filestr = f.read()
    return filestr.decode("utf-8")

def save_script(script:str, bashfile:str)->bool:
    '''
    Function that saves the script in a file.

    Parameters
    ----------
        script : str
            Script to save.
        bashfile : str
            Path of the file where the script will be saved.
    
    Returns
    -------
        bool
            True if the script was saved successfully.
    '''
    with open(bashfile,'w') as f:
        f.write(script)
    f.close()
    os.chmod(bashfile,0o777)
    return os.path.isfile(bashfile)
 
def run_bash(bashfile:str,verbose=False)->bool:
    '''
    Function that runs a bash script.

    Parameters
    ----------
        bashfile : str
            Path of the bash script.
        verbose : bool, optional
            If True, the function will print the return code of the script. The default is False.
    
    Returns
    -------
        bool
            True if the script was run successfully.
    '''

    pro = sp.run(bashfile)
    if verbose: print(pro.returncode)
    return pro.returncode

def sdss_mosaic(ra, dec, name, outdir, size=0.3, scale=0.396, band='i', sigma=True, verbose=False, keep=False):
    """
    Download a cutout image from the SDSS SAS server. 
    The image is saved in the outdir with the name of the galaxy. 

    Parameters
    ----------
        ra : float
            Right ascension of the center of the mosaic.
        dec : float
            Declination of the center of the mosaic.
        name : str
            Name of the galaxy.
        outdir : str
            Directory where the mosaic will be saved.
        size : float, optional 
            Size of the mosaic in degrees. The default is 0.3.
        scale : float, optional
            Pixel scale of the mosaic in arcseconds. The default is 0.396.
        band : str, optional
            Band of the mosaic. The default is 'i'.
        verbose : bool, optional
            If True, the function will print the URL of the script. The default is False.
        keep : bool, optional
            If True, the function will keep the intermediate files. The default is False.
    
    Returns
    -------
        bool
            True if the mosaic was downloaded successfully.
    """
    
    url = f'https://dr12.sdss.org/mosaics/script?onlyprimary=True&pixelscale={scale}&ra={ra}&filters={band}&dec={dec}&size={size}'
    if verbose: print(url)

    directory = os.path.join(outdir,name)
    bashfile = os.path.join(directory,name+'.sh')
    
    if not os.path.isdir(directory): os.mkdir(directory)

    script = sdss_script_mosaic(url)
    issaved = save_script(script,bashfile)
    
    if issaved:
        if verbose: print(f'running for {name}')
        originalpath = os.getcwd()
        os.chdir(directory)
        outcmd = run_bash(bashfile, verbose=verbose)

        if sigma:
            swarpfile = 'default.swarp'
            with open(swarpfile) as f:
                script = f.read()
            for item in script.split('\n'):
                if 'IMAGEOUT_NAME' in item:
                    script = script.replace(item, f'IMAGEOUT_NAME     {name}_sdss_sigma.fits')
                if 'WEIGHTOUT_NAME' in item:
                    script = script.replace(item, f'WEIGHTOUT_NAME    {name}_sdss_weight.fits')
        
            issaved = save_script(script, swarpfile)
            # Create sigma images for all the frames
            framelist = glob.glob('frame*.fits')
            for frame in framelist:
                img_err = get_sigma(frame)
                fits.PrimaryHDU(img_err, header=fits.getheader(frame)).writeto(frame.replace('.fits','_sigma.fits'),overwrite=True)
    
            # Combine all the sigma images
            sigmalist = glob.glob('frame*sigma.fits')
            os.system('swarp '+' '.join(sigmalist)+' -c '+swarpfile)
        
        if not keep:
            os.system(f'rm {os.path.join(directory,"frame*.fits*")}')
            os.system(f'rm {os.path.join(directory,"*.swarp")}')
        for f in glob.glob('*.fits'):
            os.system(f'mv {f} {name}-{band}.{f.split(f"-{band}.")[-1]}')
        os.chdir(originalpath)
        sleep(2)
        return outcmd==0
    else: 
        return False

def sdss_mosaic_list(row, size=0.3, scale=0.396, band='i', verbose=False):
    """
    Download an cutout image from the SDSS SAS server 
    This function is meant for parallel processing it calls
    sdss_mosaic. 

    Parameters
    ----------
        row : list
            List with the ra, dec, name and outdir of the galaxy.
        size : float, optional
            Size of the mosaic in degrees. The default is 0.3.
        scale : float, optional
            Pixel scale of the mosaic in arcseconds. The default is 0.396.
        band : str, optional
            Band of the mosaic. The default is 'i'.
        verbose : bool, optional
            If True, the function will print the URL of the script. The default is False.
    
    Returns
    -------
        bool
            True if the mosaic was downloaded successfully.
    """
    ra = row[0]
    dec = row[1]
    name = row[2]
    outdir = row[3]

    return sdss_mosaic(ra, dec, name, outdir, size=size, scale=scale, band=band, verbose=verbose)

def s4g_images(name, channel=1, mask=False, sigma=False, outdir='',verbose=False):
    """
    Download the S4G images from the IRSA server.

    Parameters
    ----------
        name : str
            Name of the galaxy.
        channel : int, optional
            Channel of the image [1->3.6, 2->4.5]. The default is 1.
        mask : bool, optional
            If True, the function will download the mask. The default is False.
        sigma : bool, optional
            If True, the function will download the sigma image. The default is False.
        outdir : str, optional
            Directory where the images will be saved. The default is ''.
        verbose : bool, optional
            If True, the function will print the URL of the images. The default is False.
    
    Returns
    -------
        bool
            True if the images were downloaded successfully.
    """
    sucess = True
    
    url = f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P4/{name}.phot.{channel}_nonan.fits'
    if verbose: print(url)
    filename = wget.download(url,out=outdir)
    sucess *= os.path.isfile(filename)
    if verbose: print(filename)

    if mask:
        url = f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P2/{name}.{channel}.final_mask.fits'
        if verbose: print(url)
        filename = wget.download(url,out=outdir)
        sucess *= os.path.isfile(filename)
        if verbose: print(filename)

    if sigma:
        url = f'https://irsa.ipac.caltech.edu/data/SPITZER/S4G/galaxies/{name}/P4/{name}_sigma2014.fits.gz'
        if verbose: print(url)
        filename = wget.download(url, out=outdir)
        sucess *= os.path.isfile(filename)
        if verbose: print(filename)

    return sucess

def legacy_mosaic(ra, dec, outdir='', name='', width=0.3, scale=0.263, bands='griz', layer='ls-dr10', verbose=False, rgb=False):
    '''
    Downloads cutotus from the Legacy Survey Server.

    Parameters
    ----------
        ra : float
            Right ascension of the center of the mosaic. [degress]
        dec : float
            Declination of the center of the mosaic. [degress]
        outdir : str, optional
            Directory where the mosaic will be saved. The default is '.'.
        name : str, optional
            Name of the output mosaic file. 
        width : float, optional
            Size of the mosaic in degrees. The default is 0.3.
        scale : float, optional
            Pixel scale of the mosaic in arcseconds. The default is 0.27.
        bands : str, optional
            Filters to be downloaded. Default is griz.
        layer : str, optional 
            Data release to use. Default is DR10 - December 2022.
        verbose : bool, optional
            If True, the function will print the URL of the script. The default is False.
    
    Returns
    -------
        bool
            True if the mosaic was downloaded successfully.
    '''
    size = np.int64(width*3600/scale)
    url = f'https://www.legacysurvey.org/viewer/fits-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&bands={bands}&size={size}'
    # url = f'https://www.legacysurvey.org/viewer/cutout.fits?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={scale}&size={round(width*3600/scale)}'
    if verbose: print(url)

    if not name: name = os.path.join(outdir,f'legacy_{ra}_{dec}_{size}_{bands}.fits')
    
    # filename = wget.download(url, out=outdir)   ## not working properly
    result = os.system(f'wget -O {name} "{url}" ')

    if rgb: 
        url = f'https://www.legacysurvey.org/viewer/jpeg-cutout?ra={ra}&dec={dec}&layer={layer}&pixscale={scale}&size={size}'
        rgbfile = name.replace('.fits','.jpg')
        # rgbfile = wget.download(url, out=outdir.replace('.fits','.jpg'))
        jpgresult = os.system(f'wget -O {rgbfile} "{url}" ')

    if verbose: print(name)
    return name


def reconstructPSF(psFieldFilename:str, filter:str,
                    row:float, col:float) -> bool:
    '''
    Computes the PSF at a given row and column 
    for a given psField structure of SDSS. 
    Code reproduce from IDL code in:
    https://www.sdss4.org/dr17/imaging/images/
    

    Parameters
    ----------
        psFieldFilename : string
            psField file name, can be an url
        
        filter : string
            Filter to compute the PSF [ugriz]

        row : int
            Row of the image to compute the PSF
        
        col : int
            Column of the image to compute the PSF
    
    Returns
    -------
        psf : numpy.ndarray
            PSF at the given row and column
    '''

    filterIdx = 'ugriz'.index(filter) + 1
    psField = fits.open(psFieldFilename)
    pStruct = psField[filterIdx].data

    nrow_b = pStruct['nrow_b'][0]
    ncol_b = pStruct['ncol_b'][0]

    rnrow = pStruct['rnrow'][0]
    rncol = pStruct['rncol'][0]

    nb = nrow_b * ncol_b
    coeffs = np.zeros(nb.size, float)
    ecoeff = np.zeros(3, float)
    cmat = pStruct['c']

    rcs = 0.001
    for ii in range(0, nb.size):
        coeffs[ii] = (row * rcs)**(ii % nrow_b) * (col * rcs)**(ii / nrow_b)

    for jj in range(0, 3):
        for ii in range(0, nb.size):
            ecoeff[jj] = ecoeff[jj] + cmat[int(ii / nrow_b), ii % nrow_b, jj] * coeffs[ii]

    psf = pStruct['rrows'][0] * ecoeff[0] + \
        pStruct['rrows'][1] * ecoeff[1] + \
        pStruct['rrows'][2] * ecoeff[2]

    psf = np.reshape(psf, (rnrow, rncol))
    psf = psf[10:41, 10:41]  # Trim non-zero regions.

    return psf

def interpolate(image, xcord, ycord, method='linear'):
    '''
    Function that replicates the IDL function INTERPOLATE 
    used to create sigma images for the SDSS survey. Specifically
    we are folloing instructions on
    https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html 

    Parameters
    ----------
        image : numpy.ndarray
            Image to interpolate.
        xcord : numpy.ndarray
            X coordinates of the image that wants to be interpolated.
        ycord : numpy.ndarray
            Y coordinates of the image that wants to be interpolated.
        method : str, optional
            Interpolation method of scipy.interpolate.griddata. 
            The default is 'linear'.
            
    Returns
    -------
        interp : numpy.ndarray
            Interpolated image.
    '''
    
    values = image.flatten()
    xx,yy = np.meshgrid(np.arange(0,image.shape[1]),
                        np.arange(0,image.shape[0]))

    new_x,new_y = np.meshgrid(xcord, ycord)
    interp = griddata(np.array([xx.flatten(),yy.flatten()]).T,
                    values,
                    (new_x,new_y),
                    method='linear')

    mask = np.where(np.isnan(interp[0,:]))[0]
    maxind = np.argmax(np.diff(mask)) + 1 
    interp[:,mask[:maxind]] = interp[:,np.max(mask[:maxind])+1,np.newaxis]
    interp[:,mask[maxind:]] = interp[:,np.min(mask[maxind:])-1,np.newaxis]

    return interp

def get_gain(framename:str)->float:
    '''
    Function that returns the gain of a given frame of SDSS
    values taken from
    https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html 

    Parameters
    ----------
        framename : str
            Name of the frame to get the gain.
    
    Returns
    -------
        gain : float
            Gain of the frame.
    '''
    table = Table(rows=[
        [1.62, 3.32, 4.71, 5.165, 4.745],
        [1.595, 3.855, 4.6, 6.565, 5.155],
        [1.59, 3.845, 4.72, 4.86, 4.885],
        [1.6, 3.995, 4.76, 4.885, 4.775],
        [1.47, 4.05, 4.725, 4.64, 3.48],
        [2.17, 4.035, 4.895, 4.76, 4.69]],
                names=['u', 'g', 'r', 'i', 'z'],)
    basename = os.path.basename(framename)

    camcol = int(basename.split('-')[3])
    filter = basename.split('-')[1]
    run = int(basename.split('-')[2])

    if (run > 1100) and (filter=='u'):
        gain = 1.825
    else:
        gain = table[filter][camcol-1]
    
    return gain

def get_darkvariance(framename:str)->float:
    '''
    Function that returns the dark variance of a given frame of SDSS
    values taken from
    https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html 

    Parameters
    ----------
        framename : str
            Name of the frame to get the dark variance.
    
    Returns
    ------- 
        darkvariance : float
            Dark variance of the frame.
    '''
    table = Table(rows=[
        [9.61, 15.6025, 1.8225, 7.84, 0.81],
        [12.6025, 1.44, 1.00, 5.76, 1.0],
        [8.7025, 1.3225, 1.3225, 4.6225, 1.0],
        [12.6025, 1.96, 1.3225, 6.25, 9.61],
        [9.3025, 1.1025, 0.81, 7.84, 1.8225],
        [7.0225, 1.8225, 0.9025, 5.0625, 1.21]],
                names=['u', 'g', 'r', 'i', 'z'],)
    camcol = int(os.path.basename(framename).split('-')[3])
    filter = os.path.basename(framename).split('-')[1]
    run = int(os.path.basename(framename).split('-')[2])
    darkvar = table[filter][camcol-1]

    if run > 1500:
        if filter=='i':
            if camcol==2: darkvar = 6.25
            if camcol==4: darkvar = 7.5625
        elif filter=='z':
            if camcol==4: darkvar = 12.6025
            if camcol==5: darkvar = 2.1025
    
    return darkvar


def get_sigma(framename:str)->np.ndarray:
    '''
    Function that returns the sigma image of a given frame of SDSS
    following the instructions on "Example of use, and calculating errors"

    https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html

    The orignal code was written in IDL and this is a python translation.

    Parameters
    ----------
        framename : str
            Name of the frame to get the sigma image.
    
    Returns
    -------
        sigma : numpy.ndarray
            Sigma image of the frame.
    '''
    img = fits.getdata(framename, 0) 

    sky = fits.getdata(framename,2)
    simg = interpolate(sky['ALLSKY'][0],
                    sky['XINTERP'][0], 
                    sky['YINTERP'][0])

    calib = fits.getdata(framename,1)
    cimg = calib[np.newaxis,:]*np.ones_like(img)

    gain = get_gain(framename)
    darkvar = get_darkvariance(framename)

    dn = img/cimg + simg
    dn_err = np.sqrt(dn/gain+  darkvar)  # in counts
    img_err = dn_err*cimg                # in nanomaggies

    return img_err


def create_sigma_psf_of_mosaic(ra, dec, name, outdir, band='i', size=0.3,
                                scale=0.396, verbose=False, keep=False):
    '''
    Function that creates the sigma and the psf images of the mosaic of a given object
    
    Parameters
    ----------
        ra : float
            Right ascension of the object.
        dec : float
            Declination of the object.
        name : str
            Name of the object.
        outdir : str
            Directory where the sigma mosaic will be saved.
        band : str
            Band of the mosaic.
        size : float
            Size of the mosaic in degrees.
        scale : float
            Pixel scale of the mosaic in arcseconds.
        verbose : bool
            If True, prints the progress of the function.
        keep : bool
            If True, keeps the downloaded files.
    
    Returns
    -------
        bool
            True if the function runs correctly.        
    '''

    # First download of all the frames
    url = f'https://dr12.sdss.org/mosaics/script?onlyprimary=True&pixelscale={scale}&ra={ra}&filters={band}&dec={dec}&size={size}'
    if verbose: print(f'getting {name} from {url}')

    directory = os.path.join(outdir,name)
    bashfile = os.path.join(directory,name+'.sh')
    
    if not os.path.isdir(directory): os.mkdir(directory)

    script = sdss_script_mosaic(url)

    # modify the script
    script = script.replace(script.split('\n')[-2],'') #remove swarp command
    for item in script.split('\n'):
        if 'IMAGEOUT_NAME' in item:
            script = script.replace(item, f'IMAGEOUT_NAME     {name}_sdss_sigma.fits')
        if 'WEIGHTOUT_NAME' in item:
            script = script.replace(item, f'WEIGHTOUT_NAME    {name}_sdss_weight.fits')
    
    issaved = save_script(script, bashfile)
    
    originalpath = os.getcwd()
    os.chdir(directory)
    if issaved:
        if verbose: print(f'running for {name}')
        outcmd = run_bash(bashfile, verbose=verbose)

    
    # Create sigma images for all the frames
    framelist = glob.glob('frame*.fits')
    for frame in framelist:
        img_err = get_sigma(frame)
        fits.PrimaryHDU(img_err, header=fits.getheader(frame)).writeto(frame.replace('.fits','_sigma.fits'),overwrite=True)
    
    # Combine all the sigma images
    sigmalist = glob.glob('frame*sigma.fits')
    os.system('swarp '+' '.join(sigmalist))


    # Find frame closest to the object to build psf
    
    pos = coords.SkyCoord(f'{ra}d {dec}d', frame='icrs')
    xid = SDSS.query_region(pos, spectro=False)

    frames = []
    for el in xid:
        file = glob.glob(f'*{el["run"]:06d}-{el["camcol"]}-{el["field"]:04d}.fits')
        for f in file:
            if os.path.exists(f):
                frames.append(f)

    frames = np.unique(frames)

    if frames.size==0:
        print(f'No frames found for {name}')

    # Build the psf for the first frame at row and col of the coordinates given
    
    hdr = fits.getheader(frames[0])
    psfpath = 'https://data.sdss.org/sas/dr12/env/PHOTO_REDUX/'
    psframe = f'psField-{hdr["RUN"]:06d}-{hdr["CAMCOL"]}-{hdr["FRAME"]:04d}.fit'
    psfurl=os.path.join(psfpath, str(hdr['RERUN']), str(hdr['RUN']),'objcs',str(hdr['CAMCOL']), psframe)

    col, row = utils.skycoord_to_pixel(coords.SkyCoord(ra, dec, frame="icrs", unit="deg"), WCS(hdr))
    row, col = float(row),float(col)
    psf = reconstructPSF(psfurl, band, row, col)

    # Adding header information

    psfhd = fits.PrimaryHDU(psf)
    psfhdu = psfhd.header
    psfhdu['URL'] = (psfpath, 'psField URL')
    psfhdu['psField'] = (psframe, 'psField file')
    psfhdu['FILE'] = (frames[0], 'Frame from where psf was computed')
    for key in ['RUN','RERUN','CAMCOL','FRAME']:
        psfhdu[key] = (np.int64(hdr[key]), f'{key} from frame file')
    psfhdu['psRA'] = (ra, 'RA of PSF center in degress')    
    psfhdu['psDEC'] = (dec, 'DEC of PSF center in degrees')
    psfhdu['psBAND'] = (band.strip(), 'Filter of PSF')
    psfhdu['psROW'] = (row, 'Row of PSF center in pixels')
    psfhdu['psCOL'] = (col, 'Column of PSF center in pixels')
    psfhdu[''] = '='*30
    psfhdu['HISTORY'] = f'File created {datetime.today()}  '
    psfhdu['COMMENT'] = psfurl
    psfhdu['COMMENT'] = 'https://www.sdss4.org/dr17/imaging/images/'

    psfhd.writeto(f'{name}_sdss_psf.fits', overwrite=True)


    if not keep:
        os.system(f'rm {os.path.join(directory,"frame*.fits*")}')
        os.system(f'rm {os.path.join(directory,"*.swarp")}')

    os.chdir(originalpath)
    
    return os.path.isfile(os.path.join(directory, f'{name}_sdss_sigma.fits')
            )*os.path.isfile(os.path.join(directory, f'{name}_sdss_psf.fits'))



def create_sigma_psf_list(params, band='i', size=0.3,
                                scale=0.396, verbose=False, keep=False):
    '''
    Function create to run parallel creation of sigma and
    psf for mosaics. It calls "create_sigma_psf_of_mosaic"
    
    Parameters
    ----------
        params: list
            List containing [ra,dec,name,outdir]
        outdir : str
            Directory where the sigma mosaic will be saved.
        band : str
            Band of the mosaic.
        size : float
            Size of the mosaic in degrees.
        scale : float
            Pixel scale of the mosaic in arcseconds.
        verbose : bool
            If True, prints the progress of the function.
        keep : bool
            If True, keeps the downloaded files.
    
    Returns
    -------
        bool
            True if the function runs correctly.        
    '''
    ra, dec, name, outdir = params
    return create_sigma_psf_of_mosaic(ra, dec, name, outdir, band=band, size=size,
                                scale=scale, verbose=verbose, keep=keep)


def mosaicPSF(ra, dec, name, outdir=None, size=0.1, scale=0.396, band='i', verbose=False):
    ''' Function that given the parameters to download a cutout from the SAS SDSS server
    downloads the PSF of the center of the cutout. 

    Parameters
    ----------
        ra : float
            Right ascension of the center of the mosaic.
        dec : float
            Declination of the center of the mosaic.
        name : str
            Name of the galaxy.
        outdir : str
            Directory where the psf will be saved.
        size : float, optional
            Size of the mosaic in degrees. The default is 0.1.
        scale : float, optional
            Pixel scale of the mosaic in arcseconds. The default is 0.396.
        band : str, optional
            Band of the mosaic. The default is 'i'.
    
    Returns
    -------
        psf : numpy.ndarray
            PSF of the center of the cutout.    
    '''
    import warnings
    if outdir is None: outdir = os.getcwd()
        
    url = f'https://dr12.sdss.org/mosaics/script?onlyprimary=True&pixelscale={scale}&ra={ra}&filters={band}&dec={dec}&size={size}'
    script = sdss_script_mosaic(url)

    pos = coords.SkyCoord(f'{ra}d {dec}d', frame='icrs')
    xid = SDSS.query_region(pos, spectro=False, data_release=12, radius='60s')
    if len(xid)==0:
        print(f'No frames found for {name}.')
        return None
    elif len(xid)==1: xid = [xid]

    frames = Table(names=xid.columns,dtype=xid.dtype)
    for el in xid:
        file = f'{el["run"]:06d}-{el["camcol"]}-{el["field"]:04d}.fits'
        if file in script: frames.add_row(el)
        else: print(f'{file} not found in SAS script for {name}.')
    if len(frames)==0: frames.add_row(xid[0])
    
    # Build the psf for the first frame at row and col of the coordinates given
    for frame in frames:
        try:
            hdr = SDSS.get_images(matches=Table(frame), band=band)[0][0].header
            psfpath = 'https://data.sdss.org/sas/dr12/env/PHOTO_REDUX/'
            psframe = f'psField-{hdr["RUN"]:06d}-{hdr["CAMCOL"]}-{hdr["FRAME"]:04d}.fit'
            psfurl = os.path.join(psfpath, str(hdr['RERUN']), str(hdr['RUN']),'objcs',str(hdr['CAMCOL']), psframe)
            col, row = utils.skycoord_to_pixel(coords.SkyCoord(ra, dec, frame="icrs", unit="deg"), WCS(hdr))
            row, col = float(row),float(col)
            psf = reconstructPSF(psfurl, band, row, col)
            pass
        except Exception as e:
            warnings.warn(f'PSF file not foud in {psfurl}.\n{e}')

    # Adding header information

    psfhd = fits.PrimaryHDU(psf)
    psfhdu = psfhd.header
    psfhdu['URL'] = (psfpath, 'psField URL')
    psfhdu['psField'] = (psframe, 'psField file')
    for key in ['RUN','RERUN','CAMCOL','FRAME']:
        psfhdu[key] = (np.int64(hdr[key]), f'{key} from frame file')
    psfhdu['psRA'] = (ra, 'RA of PSF center in degress')    
    psfhdu['psDEC'] = (dec, 'DEC of PSF center in degrees')
    psfhdu['psBAND'] = (band.strip(), 'Filter of PSF')
    psfhdu['psROW'] = (row, 'Row of PSF center in pixels')
    psfhdu['psCOL'] = (col, 'Column of PSF center in pixels')
    psfhdu[''] = '='*30
    psfhdu['HISTORY'] = f'File created {datetime.today()}  '
    psfhdu['COMMENT'] = psfurl
    psfhdu['COMMENT'] = 'https://www.sdss4.org/dr17/imaging/images/'

    psfhd.writeto(os.path.join(outdir,f'{name}_sdss_psf.fits'), overwrite=True)


    
def cross_match(catalog1, catalog2, maxsep = 2*u.arcsec):
    '''
    Given two astropy tables, cross match them and return the result.
    Each table must have ra and dec columns.

    Parameters
    ----------
        catalog1 : astropy.table.Table
            First catalog. With ra and dec columns in deg.
        catalog2 : astropy.table.Table
            Second catalog. With ra and dec columns in deg.
        maxsep : astropy.units.Quantity
            Maximum separation to consider a match.
    
    Returns
    -------
        astropy.table.Table
            Cross matched catalog.
    '''
    coords1 = coords.SkyCoord(ra=catalog1['ra'], dec=catalog1['dec'], unit=(u.deg, u.deg), frame='icrs')
    coords2 = coords.SkyCoord(ra=catalog2['ra'], dec=catalog2['dec'], unit=(u.deg, u.deg), frame='icrs')
    idx, d2d, _ = coords1.match_to_catalog_sky(coords2)
    sep_constrain = d2d < maxsep
    match1 = catalog1[sep_constrain]
    match2 = catalog2[idx[sep_constrain]]
    merge = hstack([match1,match2])
    merge['ra_1'].name = 'ra'
    merge['dec_1'].name = 'dec'
    merge.remove_columns(['ra_2','dec_2'])
    return merge

def query_gaia(ra, dec, radius, maglim=(5,25)):
    """
    Given a position and a radius, query Gaia DR3 for all sources in the field.

    Parameters
    ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        radius : float
            Radius in degrees.
        maglim : tuple, optional
            Magnitude limits to filter the query. The default is (5,25).
    
    Returns
    -------
        astropy.table.Table
            Gaia catalog.
    """
    mag_low, mag_high = maglim
    # Convert input coordinates to SkyCoord
    coords1 = coords.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    #hello
    # Query Gaia based on coordinates and radius
    query = (
        f"SELECT ra,dec,phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE CONTAINS(POINT('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec), "
        f"CIRCLE('ICRS', {coords1.ra.deg}, {coords1.dec.deg}, {radius})) = 1 "
        f"AND gaiadr3.gaia_source.phot_g_mean_mag BETWEEN {mag_low} AND {mag_high}"
    )
    job = Gaia.launch_job_async(query)
    result_table = job.get_results()
    return result_table

def query_sdss(ra, dec, width, height=None, maglim=(5,25), fields=None):
    '''
    Given a position and a radius, query SDSS DR12 for all sources in the field.
    You can set the fields to query, otherwise it will query the default fields.
    https://skyserver.sdss.org/dr12/en/help/browser/browser.aspx?cmd=description+PhotoObjAll+U#&&history=description+PhotoObjAll+U

    TODO: maybe faster with SDSS.query_sql(query)?

    Parameters
    ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        width : float
            Width of the query region in degrees.
        height : float, optional
            Height of the query region in degrees. If None, it will be set equal to width.
        maglim : tuple, optional
            Magnitude limits to filter the query. The default is (5,25).
        fields : list, optional
            Fields to query. The default is None.
    
    Returns
    -------
        astropy.table.Table
            SDSS catalog.
    '''
    if fields is None:
        photoobj_fields = ['ra','dec','modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z', 
                              'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i', 'extinction_z', 
                              'deVAB_u', 'deVAB_g', 'deVAB_r', 'deVAB_i', 'deVAB_z', 
                              'mCr4_u', 'mCr4_g', 'mCr4_r', 'mCr4_i', 'mCr4_z', 
                              'petroR50_u', 'petroR50_g', 'petroR50_r', 'petroR50_i', 'petroR50_z', 
                              'petroR90_u', 'petroR90_g', 'petroR90_r', 'petroR90_i', 'petroR90_z']
    else:
        photoobj_fields = fields
    
    if height is None:
        height = width
    
    mag_low, mag_high = maglim
    coords1 = coords.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
    tab = SDSS.query_region(coords1, width=width*u.deg, height=height*u.deg, photoobj_fields=photoobj_fields, timeout=3600)
    tab = tab[(tab['modelMag_g']>mag_low) * (tab['modelMag_g']<mag_high)]
    return tab