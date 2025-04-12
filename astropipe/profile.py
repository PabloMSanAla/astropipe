
from astropipe import utils as ut
from astropipe.plotting import show, rectangle_add_patches, ellipse_points

import numpy as np 
import os

import astropy.units as u
from astropy.io import fits
from astropy.table import QTable, Table
from astropy.wcs.utils import pixel_to_skycoord
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.visualization import ImageNormalize,LogStretch

from photutils.aperture import EllipticalAperture, EllipticalAnnulus
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.aperture import RectangularAperture

from autoprof.pipeline_steps import Isophote_Fit_FFT_Robust

import matplotlib.patches as patch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import stats
from scipy.ndimage import median_filter
from scipy.signal import medfilt, argrelextrema, savgol_filter

from scipy.interpolate import interp1d

from lmfit.models import GaussianModel

from sklearn.cluster import KMeans



class Profile:
    '''
    Class to work with surface brightness profiles. 

    Attributes
    ----------
        rad : array
            Radii of the profile in pixels
        int : array
            Intensity of the profile [in same units as image meassured]
        intstd : array
            Standard deviation of the intensity [int units]
        pa : array
            Position angle of the profile [deg]
        eps : array
            Ellipticity of the profile [1-b/a]
        center : 2d-array
            Center of the profile [pixels]
        bkg : float
            Background of the image [int units]
        bkgstd : float
            Standard deviation of the background [int units]
        zp : float
            Zero point of the image [mag]
        pixscale : float
            Pixel scale of the image [arcsec/pixel]
        mu : array
            surface brightness magnitude of the profile [mag*arcsec^-2]
        upperr : array
            Upper limit of the surface brightness magnitude [mag*arcsec^-2]
        lowerr : array
            Lower limit of the surface brightness magnitude [mag*arcsec^-2]
        columns : list
            List of the columns of the profile to be saved in table
        units : list
            List of the units of the columns
        meta : dict
            Dictionary of the metadata of the profile to be saved in table

    Methods
    -------
        set_params(radii=None, intensity=None, instensity_err=None, pa=None, eps=None, center=None, bkg=None, bkgstd=None, zp=None, pixscale=None)
            Set the parameters of the profile
        
        load(filename)
            Load the profile from a file


    '''
    def __init__(self, filename=None,  max_radius=None, init_radius=1, growth_rate=1.01):
        
        self.columns = ['radius', 'intensity', 'intensity_err', 'flux', 'flux_err',
                        'npixels','pa', 'pa_err', 'eps', 'eps_err', 'x', 'y']
        self.units = ['arcsec', 'counts', 'counts','counts', 'counts','na', 
                      'deg', 'deg', 'na', 'na', 'pixel', 'pixel']
        
        if max_radius is not None:
            alpha = np.log10(growth_rate)
            size = np.log10(max_radius)//alpha + 2
            self.rad = init_radius*10**(alpha*np.arange(0,size))

            self.int = np.zeros_like(self.rad)
            self.intstd = np.zeros_like(self.rad)
            self.flux = np.zeros_like(self.rad)
            self.fluxstd = np.zeros_like(self.rad)
            self.npixels = np.zeros_like(self.rad)
            self.pa = np.zeros_like(self.rad)
            self.pastd = np.zeros_like(self.rad)
            self.eps = np.zeros_like(self.rad)
            self.epsstd = np.zeros_like(self.rad)
        
        self.bkg = 0 
        self.bkgstd = 0 
        self.zp = 0 
        self.pixscale = 1
        self.meta = {'zp': self.zp,   'pixscale': self.pixscale, 
                     'bkg':self.bkg , 'bkgstd':   self.bkgstd}

        if filename is not None: 
            self.load(filename)
        
    def __call__(self, array, hdu=0, plot=None, save=None):
        '''
        Returns the average photometric radial profile in the array
        using the parameters of the profile.

        Parameters
        ----------
            array : str or array
                Image data or path to the fits file
            hdu : int
                HDU of the fits file. if array is a file
            plot : strs
                If given it will save the plot in the given path.
            save : str
                If given it will save the profile in the given path.
        
        Returns
        -------
            profile : astropipe.sbprofile.Profile
                Profile object with the radial profile.
        '''
        if array is str: array = fits.getdata(array, hdu)

        profile = elliptical_radial_profile(array, self.rad, (self.x, self.y), self.pa, self.eps, 
                                    plot=plot, save=save)
        
        profile.set_params(bkg=self.bkg, bkgstd=self.bkgstd, 
                           zp=self.zp, pixscale=self.pixscale,
                           pastd=self.pastd, epsstd=self.epsstd)
        profile.brightness()

        return profile
    
    def set_params(self, 
            radii=None, intensity=None, instensity_err=None, 
            flux=None, fluxstd=None, npixels=None,
            pa=None, pastd=None, eps=None, epsstd=None, center=None, 
            bkg=None, bkgstd=None, zp=None, pixscale=None):
        '''
        Sets the parameters of the profile.
        '''
        
        if radii is not None: self.rad = radii
        if intensity is not None: self.int = intensity
        
        # This conversion variable is only to create arrays of static pa, eps, and center
        if np.size(pa) == np.size(eps) == 1:
            conversion = np.ones_like(self.rad)
            self.type = 'fixed'
        else: 
            conversion, self.type = 1, 'dynamic'
        
        if instensity_err is not None: self.intstd = instensity_err
        
        if pa is not None: self.pa = pa*conversion
        if pastd is not None: self.pastd = pastd*conversion
        if eps is not None: self.eps = eps*conversion
        if epsstd is not None: self.epsstd = epsstd*conversion
        if center is not None and np.array(center).size==2: 
            self.x = center[0]*conversion
            self.y = center[1]*conversion
        elif center is not None: self.x, self.y = center
        
        if flux is not None: self.flux = flux*conversion
        if fluxstd is not None: self.fluxstd = fluxstd*conversion
        if npixels is not None: self.npixels = npixels*conversion

        if bkg is not None: self.bkg = bkg
        if bkgstd is not None: self.bkgstd = bkgstd 
        if zp is not None: self.zp = zp
        if pixscale is not None: self.pixscale = pixscale

        self.meta = {'zp':zp, 'pixscale':pixscale, 'bkg':bkg , 'bkgstd':bkgstd}

        if zp is not None and pixscale is not None and any(self.int>0):
            self.brightness()
        
    
    def brightness(self, zp=None, bkg=None, pixscale=None,  bkgstd=None):
        '''Computes the surface brightness of the profile
        from the intensity and the background. It also computes
        the upper and lower limits of the profile.
        
        Parameters
        ----------
            zp : float
                Zero point of the image [mag]
            bkg : float
                Background of the image [int units]
            pixscale : float
                Pixel scale of the image [arcsec/pixel]
            bkgstd : float
                Standard deviation of the background [int units]
        
        Returns
        -------
            mu : array
                surface brightness magnitude of the profile [mag*arcsec^-2]
            upperr : array
                Upper limit of the surface brightness magnitude [mag*arcsec^-2]
            lowerr : array
                Lower limit of the surface brightness magnitude [mag*arcsec^-2]
        '''

        zp = self.zp if zp is None else zp
        bkg = self.bkg if bkg is None else bkg
        bkgstd = self.bkgstd if bkgstd is None else bkgstd
        pixscale = self.pixscale if pixscale is None else pixscale
        
        self.mu, self.upperr, self.lowerr = get_surface_brightness(
            self.rad, self.int,  self.intstd, bkg, bkgstd, pixscale, zp)
        
    def skycenter(self, WCS):
        ''' Given an WCS object, 
        it computes the sky coordinates of the center of the profile
        and sets the ra and dec attributes.
        
        Parameters
        ----------
            WCS : astropy.wcs.WCS
                WCS object of the image
        '''
        self.ra, self.dec = pixel_to_skycoord(self.x, self.y, WCS)
    
    def plot(self, axes=None, color='r', label=None, **kwargs):
        '''Plots the radial profile of the galaxy.
        It uses the plot_profile function. 
        
        Parameters
        ----------
            axes : tuple of matplotlib.axes
                Axes to plot the profile, eps and pa
            color : str
                Color of the profile
            label : str
                Label of the profile
            **kwargs : dict
                Additional arguments to pass to the plot
        
        Returns
        -------
            fig : matplotlib.figure
                Figure with the profile and the parameters'''
        label = self.type if label is None else label
        fig = plot_profile(self.rad*self.pixscale, self.mu, self.pa, self.eps, self.upperr, self.lowerr, 
                           axes=axes, color=color, label=label, **kwargs)
        return fig

    def extend(self, data, max_radius, growth_rate=None):
        '''Extends the radial profile to a new maximum radius
        using the growth rate of the profile.
        
        Parameters
        ----------
            data : array
                Image data where the profile was extracted
            max_radius : float
                New maximum radius of the profile
            growth_rate : float
                Growth rate of the profile. If None it will use the median growth rate.
        
        Returns
        -------
            None
        '''

        if growth_rate is None: growth_rate = np.nanmedian(self.rad[1:]/self.rad[:-1])
        alpha = np.log10(growth_rate)
        n = np.log10(max_radius/self.rad[-1])//alpha + 1
        new_rad = self.rad[-3]*10**(alpha*np.arange(1,n+1))
        new_prof = elliptical_radial_profile(data, new_rad, (self.x[-1], self.y[-1]), self.pa[-1], self.eps[-1])
        self.rad = np.concatenate((self.rad, new_rad[2:]))
        self.int = np.concatenate((self.int, new_prof.int[2:]))
        self.intstd = np.concatenate((self.intstd, new_prof.intstd[2:]))
        self.pa = np.concatenate((self.pa, new_prof.pa[2:]))
        self.pastd = np.concatenate((self.pastd, new_prof.pastd[2:]))
        self.eps = np.concatenate((self.eps, new_prof.eps[2:]))
        self.epsstd = np.concatenate((self.epsstd, new_prof.epsstd[2:]))
        self.x = np.concatenate((self.x, new_prof.x[2:]))
        self.y = np.concatenate((self.y, new_prof.y[2:]))
        self.flux = np.concatenate((self.flux, new_prof.flux[2:]))
        self.fluxstd = np.concatenate((self.fluxstd, new_prof.fluxstd[2:]))
        self.brightness()
    
    def remove_nans(self):
        if not hasattr(self, 'mu'): self.brightness()
        half = len(self.mu)//2
        cutind = half + np.argwhere(np.isnan(self.mu[half:]))[0][0]
        self.rad = self.rad[:cutind]
        self.int = self.int[:cutind]
        self.intstd = self.intstd[:cutind]
        self.flux = self.flux[:cutind]
        self.fluxstd = self.fluxstd[:cutind]
        self.npixels = self.npixels[:cutind]
        self.pa = self.pa[:cutind]
        self.pastd = self.pastd[:cutind]
        self.eps = self.eps[:cutind]
        self.epsstd = self.epsstd[:cutind]
        self.x = self.x[:cutind]
        self.y = self.y[:cutind]
        self.brightness()
    
    def interpolateCurve(self, var1, var2, nElements=10000, kind='linear'):
        '''
        Enhances resolution of the curve of growth via interpolation.

        Parameters
        ----------
        nElements : int
            Number of resolution elements for interpolated curve
        kind : string
            Interpolation type.  Valid values are: ‘linear’, ‘nearest’,
            ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            or ‘next’.

        Returns
        -------
        high_res_var1 : numpy.ndarray
            Variable 1 with resolution defined by nElements
        high_res_var2 : numpy.ndarray
            Variable 2 with resolution defined by nElements
        '''
        f = interp1d(var1, var2, kind=kind)
        high_res_var1 = np.linspace(np.min(var1), np.max(var1), nElements)
        high_res_var2 = f(high_res_var1)

        return high_res_var1, high_res_var2

    def curveOfGrowth(self, sky=None):
        '''
        Produces a curve of growth in magnitudes

        Parameters
        ----------
        sky : float, optional
            Estimate of the average sky flux local to the galaxy.
            The default is 0.

        Returns
        -------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma
        '''
        if sky is None: sky = self.bkg

        sma = self.rad
        tflux = self.flux - self.npixels*sky
        mags = -2.5*np.log10(tflux) + self.zp

        return sma, mags

    def totalMagnitude(self, sma, mags, npoints=10):
        '''
        Computes total magnitude of galaxy using curve of growth

        Parameters
        ----------
        sma : numpy.ndarray
            Semi-major-axis array, in pixels
        mags : numpy.ndarray
            Total magnitude enclosed within sma
        npoints : int, optional
            Number of points to use for fitting. The default is 10.

        Returns
        -------
        totalMag : float
            Total magnitude extrapolated to infinity

        NOTE: fitting limits currently hard-coded, but proven effective for
        exponential profiles
        '''
        slope = ut.localSlope(sma, mags)
        # want = (slope >= -0.008) & (slope <= 0)  # Typical useful range
        want = (sma>sma[-npoints])
        fit = np.polyfit(slope[want], mags[want], 1)
        totalMag = fit[1]

        return totalMag

    def fractionalRadius(self, totalMag, fluxFrac=0.5):
        '''
        Computes the radius containing some fraction of the galaxy's total
        light.

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        fluxFrac : float, optional
            The desired fraction of enclosed light. The default is 0.5, i.e.
            by defaul this returns the half-light radius.

        Returns
        -------
        fracRad : float
            The radius containing the desired fraction of the galaxy's total
            light, in arcseconds
        '''
        totalFlux = 10**(-0.4*(totalMag - self.zp))
        findFlux = totalFlux * fluxFrac

        # A little crude, but we interpolate to improve the radius resolution
        sma, cog = self.interpolateCurve(self.rad,
                                         self.flux-self.npixels*self.bkg)

        idx = ut.closest(cog, findFlux)
        fracRad = sma[idx]

        return fracRad*self.pixscale

    def concentration(self, totalMag, f1 = 0.8, f2=0.2):
        '''
        Derives the concentration parameter C discussed by Conselice (2003)
        between fraction f1 and f2. Default values are the concentration index
        originally proposed by Kent (1985), ApJS, 59, 115. [C82]

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        f1 : float, optional
            Outer fraction level. The default is 0.8. [0-1]
        f2 : float, optional
            Inner fraction level. The default is 0.2. [0-1]

        Returns
        -------
        c82 : float
            The concentration parameter C = 5log(R_f1/R_f2)
        '''
        r1 = self.fractionalRadius(totalMag, f1)
        r2 = self.fractionalRadius(totalMag, f2)
        c = 5*np.log10(r1/r2)
        return c

    def isophotalRadius(self, SbMu, sky=None, returnMorph=False):
        '''
        Get radius at a level of surface brightness

        Parameters
        ----------
            SbMu : float
                Surface brightness level
        Returns
        -------
            rad : float
                Radius at the given surface brightness level [arcseconds]
        '''

        if not hasattr(self, 'mu') and not sky: 
            self.brightness()
    
        if sky: 
            mu = self.zp - 2.5*np.log10(self.int - sky) + 5*np.log10(self.pixscale)
        else: mu = np.zeros_like(self.mu) + self.mu

        sma, sb = self.interpolateCurve(self.rad,
                                        mu)
        if returnMorph: 
            _, pa = self.interpolateCurve(self.rad, self.pa)
            _, eps = self.interpolateCurve(self.rad, self.eps)
    
        idx = ut.closest(sb, SbMu)
        output = sma[idx]*self.pixscale
        if returnMorph: output = [output, pa[idx], eps[idx]]

        return output

    def concentrationRe(self, totalMag, alpha=0.7):
        '''
        Derived the concentration parameter proposed by Trujillo et al. (2001)
        MNRAS, 326, 869

        Parameters
        ----------
        totalMag : float
            Galaxy total magnitude
        alpha : float
            Factor by which to define outer isophote level, as alpha*rEff

        Returns
        -------
        cRe : float
            The concentration parameter sum(I[<alpha*Reff])/sum(I[<Reff])
        '''
        rEff = self.fractionalRadius(totalMag, 0.5)

        sma, cog = self.interpolateCurve(self.rad,
                                         self.flux-self.npixels*self.bkg)

        idx = ut.closest(sma, rEff)
        idy = ut.closest(sma, alpha*rEff)

        sum1 = cog[idy]
        sum2 = cog[idx]

        cRe = sum1/sum2

        return cRe

    def petrosianRadius(self, eta=0.2, sky=0):
        '''
        Derives Petrosian radius (Petrosian 1976).

        Parameters
        ----------
        eta : float, optional
            Value of Petrosian index used to define Petrosian radius.
            The default is 0.2 (Bershady et al. 2000)

        Returns
        -------
        radPetro : float
            Petrosian radius in arcseconds

        NOTE: quick check, for an exponential profile, R_petrosian ~
        2x R_eff.
        '''
        if sky==0: sky = self.bkg

        sma, sb = self.interpolateCurve(self.rad,
                                        self.int-sky)
        __, cog = self.interpolateCurve(self.rad,
                                        self.flux-self.npixels*sky)
        __, area = self.interpolateCurve(self.rad,
                                         self.npixels)
        petrosian = sb * (area/cog)
        idx = ut.closest(petrosian, eta)
        radPetro = sma[idx]

        return radPetro*self.pixscale
    
    def surfaceBrightness(self, radius, sky=None):
        '''Computes the surface brightness level at a given radius
        
        Parameters
        ----------
            radius : float
                Radius of the surface brightness level [arcsec]
            sky : float
                Sky background of the image [default is None]

        Returns
        -------
            mu : float
                Surface brightness level at the given radius [mag*arcsec^-2]
        '''
        if sky is not None:
            mu = self.zp - 2.5*np.log10(self.int - sky) + 5*np.log10(self.pixscale)
        else:
            mu = self.mu
        sma, sb = self.interpolateCurve(self.rad,
                                        mu)
        idx = ut.closest(sma, radius/self.pixscale)
        return sb[idx]

    def averageSurfaceBrightness(self, radius, sky=None):
        '''Computes the average surface brightness level 
        inside  a given radius 
        
        Parameters
        ----------
            radius : float
                Maximum radius used to measeure average SB [arcsec]
            sky : float
                Sky background of the image [default is None]

        Returns
        -------
            mu : float
                Average surface brightness level at the given radius [mag*arcsec^-2]
        '''
        if sky is not None:
            mu = self.zp - 2.5*np.log10(self.int - sky) + 5*np.log10(self.pixscale)
        else:
            mu = self.mu
        sma, sb = self.interpolateCurve(self.rad,
                                        mu)
        idx = ut.closest(sma, radius/self.pixscale)
        return np.nanmean(sb[:idx])


    def write(self, filename=None, overwrite=True):
        '''Saves the profile in a fits file'''
        self.meta = {'zp':self.zp, 'pixscale': self.pixscale, 
                     'bkg':self.bkg , 'bkgstd': self.bkgstd}
        
        if filename is None: filename = 'profile.txt'
        
        table = Table([self.rad, self.int, self.intstd, 
                       self.flux, self.fluxstd, self.npixels,
                       self.pa, self.pastd, self.eps, self.epsstd,
                       self.x, self.y], 
                        names=self.columns,
                        units=self.units, meta=self.meta)
        table.write(filename, overwrite=overwrite)
        return os.path.isfile(filename)
    
    def load(self, filename):
        '''Loads previously saved profile from file'''
        table = Table.read(filename)
        self.set_params(np.array(table[self.columns[0]].value), 
            np.array(table[self.columns[1]].value), np.array(table[self.columns[2]].value), 
            np.array(table[self.columns[3]].value), np.array(table[self.columns[4]].value), 
            np.array(table[self.columns[5]].value), 
            np.array(table[self.columns[6]].value), np.array(table[self.columns[7]].value), 
            np.array(table[self.columns[8]].value), np.array(table[self.columns[9]].value),
            (np.array(table[self.columns[10]].value), np.array(table[self.columns[11]].value)), 
            table.meta['BKG'], table.meta['BKGSTD'], table.meta['ZP'], table.meta['PIXSCALE'])
        self.table = table

    def load_isolist(self, isolist):
        '''Loads the profile from an photutils.isophote.IsophoteList object'''
        self.set_params(isolist.sma, isolist.intens, isolist.int_err, 
                        isolist.tflux_e, isolist.npix_e,
                        isolist.pa*180/np.pi, isolist.pa_err*180/np.pi,
                        isolist.eps, isolist.ellip_err, (isolist.x0, isolist.y0))
        self.brightness()

    def load_isolist_table(self, tableFile,format=None):
        '''Loads the profile from a table file with 
        same format as a photutils.isophote.IsophoteList'''
        tbl = Table.read(tableFile, format=format)
        self.set_params(radii=tbl['sma'].value, intensity=tbl['intens'].value, instensity_err=tbl['int_err'].value, 
            flux=tbl['tflux_e'].value, fluxstd=None, npixels=tbl['npix_e'].value,
            pa=tbl['pa'].value-90, pastd=tbl['pa_err'].value, eps=tbl['eps'].value, epsstd=tbl['ellip_err'].value, 
            center=(tbl['x0'].value,tbl['y0'].value))

def plot_profile(radius, mu, pa, eps, mupper=None, mlower=None, axes=None, color='k', label=None, **kwargs):
    '''
    Function to plot the surface brightness profile of a galaxy
    and the elliptical parameters used in the photometry (eps and pa).
    It also plots the upper and lower limits of the profile.

    Parameters
    ----------
        radius : array
            Radius of the profile [arcsec]
        mu : array
            Surface brightness of the profile [mag*arcsec^-2]
        pa : float
            Position angle of the galaxy [deg]
        eps : float
            Ellipticity of the galaxy 
        mupper : array
            Upper error of the surface brightness [mag*arcsec^-2]
        mlower : array
            Lower error of the surface brightness [mag*arcsec^-2]
        ax : tuple of matplotlib.axes
            Axes to plot the profile, eps and pa
        color : str
            Color of the profile
        label : str
            Label of the profile
        **kwargs : dict
            Additional arguments to pass to the plot function
    
    Returns
    -------
        fig : matplotlib.figure
            Figure with the profile and the parameters
    '''

    if axes is None: 
        fig = plt.figure(figsize=(5,6))
        axeps = plt.subplot2grid((5,1),(4,0))
        axmu = plt.subplot2grid((5,1),(0,0),rowspan=3,sharex=axeps)
        axpa = plt.subplot2grid((5,1),(3,0),sharex=axeps)
    else:
        axmu, axpa, axeps = axes
        fig = axmu.get_figure()

    # Surface Brightness plot
    axmu.plot(radius, mu, color=color, label=label, **kwargs)
    if mupper is not None and mlower is not None: 
        axmu.fill_between(radius, mupper, mlower, color=color, alpha=0.3)
    elif mupper is not None: axmu.plot(radius, mupper, color=color, ls='--')
    elif mlower is not None: axmu.plot(radius, mlower, color=color, ls='--')
    
    axmu.set_ylabel('$\mu\,[\mathrm{mag\,arcsec}^{-2}]$',fontsize=14,labelpad=3)
    axmu.invert_yaxis()

    # Positional Angle plot
    pa[pa>180] = pa[pa>180] - 180
    pa[pa<0] = pa[pa<0] + 180
    axpa.plot(radius, pa, color=color, **kwargs)
    axpa.set_ylabel('PA (deg)',fontsize=12, labelpad=3)
    axpa.set_ylim([-10,190]); axpa.set_yticks([0,45,90,135,180])
    axpa.locator_params(axis='y',nbins=3, tight=True)
    axpa.tick_params(labelsize=12)
    #axpa.yaxis.set_label_position('right')

    # Epsilon plot
    axeps.plot(radius, eps, color=color, **kwargs)
    axeps.set_ylabel('$\\epsilon$',fontsize=12,labelpad=3)
    axeps.set_xlabel('Radius (arcsec)',fontsize=14,labelpad=3)
    axeps.set_ylim([-0.1,1.1]); axeps.set_yticks([0,0.5,1])
    axeps.tick_params(labelsize=12)

    for ax in [axmu, axpa]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [axmu,axpa,axeps]:
        ax.grid(ls='--',alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15)
    return fig

def get_surface_brightness(rad, intensity,  intensitystd, bkg, bkgstd, pixscale, zp):
    '''
    Function to calculate the surface brightness from the intensity.
    It also computes the lower and upper uncertanties due to the 
    background and intensity std. It interpolates the nan values. 
    '''
    mag = zp -2.5*np.log10(intensity - bkg) + 5*np.log10(pixscale)
    lowerr = zp - 2.5*np.log10(intensity - bkg + intensitystd + bkgstd) + 5*np.log10(pixscale)
    upperr = zp - 2.5*np.log10(intensity - bkg - intensitystd - bkgstd) + 5*np.log10(pixscale)
    if any(np.isnan(lowerr)): lowerr[np.isnan(lowerr)] = np.interp(rad[np.isnan(lowerr)], rad[~np.isnan(lowerr)], lowerr[~np.isnan(lowerr)])
    if any(np.isnan(upperr)): 
        ind = np.argwhere(np.isnan(upperr))
        # slope = np.nanmedian(mag[ind]-mag[ind-1])
        # upperr[ind] =  upperr[ind[0]-1] + slope*(rad[ind] - rad[ind[0]-1])
        maxdiff = np.nanmax(upperr-mag)
        upperr[ind] = mag[ind] + maxdiff
        # if any(np.isnan(upperr)):
        #     upperr[np.isnan(upperr)] = np.interp(rad[np.isnan(upperr)], rad[~np.isnan(upperr)], upperr[~np.isnan(upperr)])
    maxdiff = np.nanmax([np.abs(mag-lowerr), np.abs(mag-upperr)])
    upperr[upperr<mag] = mag[upperr<mag] + maxdiff
    lowerr[lowerr>mag] = mag[lowerr>mag] - maxdiff
    return mag, upperr, lowerr


def elliptical_radial_profile(data, rad, center, pa, eps, growth_rate=1.03, weight=None,
                        plot=None, save=None):
    '''
    Sigma clipped average elliptical radial profile of a galaxy.
    It uses the elliptical apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        rad : array or float
            Radius of the apertures or maximum radius of aperture.
        center : tuple
            Center of the galaxy (x,y).
        pa : float
            Position angle of the galaxy (degrees).
        eps : float
            Ellipticity of the galaxy [1-b/a]
        max_r : float
            Maximum radius of the profile.
        growth_rate : float
            Growth rate of the apertures.
        weight : array
            Weight of the pixels to do an aritmetic mean.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : astropipe.sbprofile.Profile
            Profile object with the radial profile.
   '''

    if type(rad) not in [np.ndarray, list]:
        profile = Profile(max_radius=rad, growth_rate=growth_rate)
        profile.set_params(pa=pa, eps=eps, center=center)
    else:
        profile = Profile()
        profile.set_params(pa=pa, eps=eps, center=center, radii=rad,
                           intensity=np.zeros_like(rad), instensity_err=np.zeros_like(rad),
                           flux=np.zeros_like(rad), fluxstd=np.zeros_like(rad), 
                           npixels=np.zeros_like(rad),
                           pastd=np.zeros_like(rad), epsstd=np.zeros_like(rad))    
           
    if weight is None: weight = np.ones_like(data)

    # TODO: If keyword rad is givem maybe extrapolate pa, and eps¿?
    # if rad is not None:
    #     profile.pa = np.interp(rad, profile.rad, profile.pa)
    #     profile.eps = np.interp(rad, profile.eps, profile.eps)
    #     profile.set_params(center=(center[0], center[1]))
    #     profile.center = np.interp(rad, profile.rad, profile.center)

    ellip_apertures = []

    previous_mask = np.zeros_like(data)

    for i,rad in enumerate(profile.rad):
        if i>0:
            profile.rad[i] = (profile.rad[i-1] + profile.rad[i])/2
        # generate astropy aperture
        ellip_apertures.append(EllipticalAperture(
            (profile.x[i],profile.y[i]), rad, 
            (1-profile.eps[i])*rad, profile.pa[i]*np.pi/180))
        
        # create mask and index list of the aperture and updates previous one 
        mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)
        index = (data.mask==False) * (mask!=0) * (previous_mask==0)
        maskinside = (data.mask==True)*(mask!=0)*(previous_mask==0)
        previous_mask = mask

        # compute sigma clipped median of the aperture 
        clipped = sigma_clip(data.data[index], sigma=2.5, maxiters=3)
        norm = np.sum(weight[index][clipped.mask==False])
        profile.int[i] = np.ma.sum(clipped*weight[index]) / norm
        profile.intstd[i] = np.nanstd(clipped)/np.sqrt(np.size(clipped))
        if np.isnan(profile.intstd[i]): profile.intstd[i] = profile.intstd[i-1] 
        
        # integrated photometry 
        profile.npixels[i] =  profile.npixels[i-1] + np.size(data.data[index]) + np.sum(maskinside)
        profile.flux[i] =  profile.flux[i-1] + np.nansum(data.data[index]) + np.nanmax([0,profile.int[i]*np.sum(maskinside)])
        profile.fluxstd[i] = np.sqrt(profile.fluxstd[i-1]**2 +  (np.sqrt(profile.npixels[i-1])*profile.intstd[i-1])**2)

    if plot is not None:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for ap in ellip_apertures:
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]],
                            [ax.set_xlim, ax.set_ylim]):
            lim([val-1.1*profile.rad[-1], val+1.1*profile.rad[-1]])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save.split('.')[-2]+'_static.fits',overwrite=True)

    return profile


def isophotal_photometry(data, center, pa, eps, reff,  max_r=None, growth_rate=1.03,
                        fix_center=False, fix_pa=False,fix_eps=False, plot=None, save=None):
    '''
    Sigma clipped average radial profile of a galaxy.
    It uses the elliptical apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        center : tuple
            Center of the galaxy.
        pa : float
            Guess of the position angle of the galaxy.
        eps : float
            Guess ellipticity of the galaxy [1-b/a]
        max_r : float
            Maximum radius of the profile.
        growth_rate : float
            Growth rate of the apertures.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : astropipe.sbprofile.Profile
            Profile object with the radial profile.
   '''
    pa = pa * np.pi/180
    step = growth_rate - 1
    
    guess_aper = EllipseGeometry(x0=center[0], y0=center[1],
                            sma=reff, eps=eps, pa=pa)

    ellipse = Ellipse(data, guess_aper)

    isolist = ellipse.fit_image(reff, integrmode='median',sclip=2.5, nclip=3, maxsma=max_r,
                minsma=1,step=step,fix_center=fix_center,fix_pa=fix_pa,fix_eps=fix_eps)
    
    fluxstd = np.sqrt(isolist.npix_e)*isolist.int_err

    profile = Profile()
    profile.set_params(radii=isolist.sma, intensity=isolist.intens, instensity_err=isolist.int_err,
        flux=isolist.tflux_e, fluxstd=fluxstd, npixels=isolist.npix_e,
        pa=isolist.pa*180/np.pi, pastd=isolist.pa_err*180/np.pi,
        eps=isolist.eps, epsstd=isolist.ellip_err, center=(isolist.x0, isolist.y0))
    
    if plot is not None:
        fig, ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for i,rad in enumerate(profile.rad):
            ap = EllipticalAperture(
            (profile.x[i], profile.y[i]), rad, 
            (1-profile.eps[i])*rad, profile.pa[i]*np.pi/180)
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]], 
                            [ax.set_ylim, ax.set_xlim]):
            lim([val-1.1*max_r,val+1.1*max_r])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save, overwrite=True)

    return profile


def rectangular_radial_profile(data, rad, center, pa, width=2, growth_rate=1.01, 
                        plot=None, save=None):
    
    '''
    Sigma clipped average rectangular radial profile of a galaxy.
    It uses the rectangular apertures to get the radial profile.
    
    Parameters
    ----------
        data : 2D array
            Image data.
        rad : array or float
            Radius of the apertures or maximum radius of aperture.
        center : tuple
            Center of the galaxy (x,y).
        pa : float
            Position angle of the galaxy.
        widht : float
            Width of the rectangular apertures.
        growth_rate : float
            Growth rate of the apertures.
        plot : str
            If given it will save the plot in the given path.
        save : str
            If given it will save the profile in the given path.
    
    Returns
    -------
        profile : astropipe.sbprofile.Profile
            Profile object with the radial profile.
   '''

    if type(rad) not in [np.ndarray, list]:
        profile = Profile(max_radius=rad, growth_rate=growth_rate)
        profile.set_params(pa=pa, eps=0, center=center)
    else:
        profile = Profile()
        profile.set_params(radii=rad, pa=pa, eps=0, center=center,
                           intensity=np.zeros_like(rad), instensity_err=np.zeros_like(rad))

    # TODO: instead of fixed width apertures use increasingly large apertures to improve S/N at large radii


    rect_apertures, std = [], 0 

    previous_mask = np.zeros_like(data)

    for i,rad in enumerate(profile.rad):

        if len(rect_apertures) > 1:
            previous_mask = mask
            std = np.nanmedian(profile.intstd[-1])

        rect_apertures.append(RectangularAperture(
            (profile.x[i],profile.y[i]), 2*rad, width, pa*np.pi/180))
        mask = rect_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = (data.mask==False) * (mask!=0) * (previous_mask==0)
        clipped = sigma_clip(data.data[index], sigma=2.5, maxiters=3)
        profile.int[i] = np.ma.median(clipped)
        profile.intstd[i] = np.nanstd(clipped)/np.sqrt(np.size(clipped))
        
        # if np.nanmedian(profile.intstd[-1]) > 3*std:
        #     width *= 1.5
        #     std = np.nanmedian(profile.intstd[-1])

    if plot is not None:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        show(data, ax=ax)
        for ap in rect_apertures:
            ap.plot(color='black', alpha=0.6,lw=0.5, axes=ax)
        for val,lim in zip([profile.x[0], profile.y[0]],
                            [ax.set_xlim, ax.set_ylim]):
            lim([val-1.1*profile.rad[-1], val+1.1*profile.rad[-1]])
        fig.savefig(plot, dpi=200, bbox_inches='tight', pad_inches=0.1)

    if save is not None:
        profile.write(save.split('.')[-2]+'_rectangular.fits',overwrite=True)

    return profile


def rectangular_photometry(IMG, center=None, pa=None, width=5,
                       zp=None, growth_rate = 1.03, max_r = 1000,plot=False,save=None):
    
    if not pa: pa =  IMG.pa * np.pi/180
    if not center: center = IMG.pix
    if not zp: zp = IMG.zp

    
    data = IMG.data - IMG.bkg 

    rect_apertures = []
    intensity_rec = []
    intensity_rec_std = []
    radii = []
    previous_mask = np.zeros_like(IMG.data)

    if plot:
        fig = plt.figure()
        ax = plt.subplot(111)
        vmin = IMG.mu_to_counts(IMG.maglim)
        vmax = IMG.mu_to_counts(18)

        norm = ImageNormalize(IMG.data,vmin=vmin,vmax=vmax,stretch=LogStretch())
        
        im = ax.imshow(data,norm=norm,interpolation='none', 
                        origin='lower',cmap='nipy_spectral_r')
        
        bar = fig.colorbar(im,ticks=IMG.mu_to_counts(np.arange(18,IMG.maglim,2.5)))
        bar.set_ticklabels(np.arange(18,IMG.maglim,2.5))
    
    sma = 1
    while sma <= 2*max_r:


        if len(rect_apertures) > 1:
            previous_mask = mask

        rect_apertures.append(RectangularAperture((center[0],center[1]), sma, width, pa))
        mask = rect_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = ut.where([data.mask==False,mask!=0,previous_mask==0])
        intensity_rec.append(np.ma.median(data.data[index]))
        intensity_rec_std.append(np.ma.std(data.data[index])/np.sqrt(
                                np.size(data.data[index])))
        radii.append(sma/2)
        sma *= growth_rate

        if plot:
            rect_apertures[-1].plot(color='black',alpha=0.7,lw=0.7,axes=ax)
    
    if plot: 
        ax.set_xlim([IMG.pix[0]-max_r,IMG.pix[0]+max_r])
        ax.set_ylim([IMG.pix[1]-max_r,IMG.pix[1]+max_r])
        plt.tight_layout()


    sb_profile  = zp - 2.5*np.log10(np.divide(intensity_rec,IMG.pixel_scale**2))
    lower_err   = zp - 2.5*np.log10(np.add(intensity_rec,intensity_rec_std)/IMG.pixel_scale**2) 
    upper_err   = zp - 2.5*np.log10(np.subtract(intensity_rec,intensity_rec_std)/IMG.pixel_scale**2) 
    
    lower_err = sb_profile - lower_err 
    upper_err = upper_err - sb_profile  
    
    lower_err[np.isnan(lower_err)] = np.nanmin(lower_err)
    upper_err[np.isnan(upper_err)] = np.nanmax(upper_err) 

    radii = np.array(radii)*IMG.pixel_scale

    profile = QTable([radii*u.arcsec ,sb_profile * (u.mag / u.arcsec**2),
                    lower_err*u.mag/u.arcsec**2,upper_err*u.mag/u.arcsec**2],
                    names=['radius','surface_brightness','sb_err_low','sb_err_upper'],
                    meta = {'pixel_scale':IMG.pixel_scale,
                    'zero_point':zp,
                    'pa':pa,'width':width,
                    'x0':center[0],'y0':center[1]})
    
    if save:
        profile.write(save.profile.split('.')[-2]+'_rect.fits',overwrite=True)
        if plot:
            plt.savefig(os.path.join(save.temp,
                os.path.basename(save.profile).split('.')[-2]+'_rectangles.jpg'),dpi=200)

    return profile

def random_rectangular_boxes(center, pa, sma, eps, n=5, wbox=30):
    ecc = np.sqrt(1-(1-eps)**2)
    omegas = np.linspace(0,1.9*np.pi,n)
    r = (sma*(1-eps))/np.sqrt(1-(ecc*np.cos(omegas))**2)
    dx = r*np.cos(omegas)*(1+np.random.randint(2,10,n)/100)
    dy = r*np.sin(omegas)*(1+np.random.randint(2,10,n)/100)
    dx,dy = ut.change_coordinates(np.array([dx,dy]),np.array((0,0)),pa)
    centers = np.array([center[0]+dx,center[1]+dy])
    
    rectangles = RectangularAperture(centers.T,wbox, wbox)
    return rectangles

def background_estimation_euclid(data, center, pa, eps, init=1, growth_rate=1.04, seed=1234323, plot=None):
    ''' Estimate background'''

    rng = np.random.default_rng(seed)

    # Get major axes coordinates at each pixel
    smacoord = create_ellipse_meshgrid(eps, pa, center, data.shape)
    x = smacoord.flatten()[~data.mask.flatten()]
    y = data.flatten()[~data.mask.flatten()]

    # Create radial bines
    alpha = np.log10(growth_rate)
    size = np.log10(np.nanmax(data.shape))//alpha + 2
    rad = 10**(alpha*np.arange(0,size))

    # Get intensity profile
    _,intensity,intensity_err = sigma_clipped_stats(create_matrix_by_bins(x,y,rad),axis=1)
    intsmooth = savgol_filter(intensity, 10, 1)
    sma = (rad[:-1]+rad[1:])/2
    
    # Get derivative and background radius
    n_sigma = 3
    dIdr = ut.derivative(sma,intensity)
    exp_crit = np.abs(intensity/dIdr)
    exp_crit_smooth = np.abs(intsmooth/ut.derivative(sma,intsmooth))
    n= np.log10(np.nanmax(sma))//1
    exp_crit_norm = (10**n)*exp_crit_smooth/(10**n+sma)
    _,med_exp,std_exp = sigma_clipped_stats(exp_crit_norm)
    bkgrad = sma[(exp_crit_norm>med_exp+n_sigma*std_exp)*(sma>init)][0]//1

    # Measure background value around an elliptical annnulai
    scale = 1.1
    sky_ellip_aper = data[(smacoord>bkgrad)*(smacoord<=scale*bkgrad)*(~data.mask)].flatten()

    # Measure the background using the elliptical annulus
    _, gauss_fit = find_mode(sky_ellip_aper)  
    ellip_bkg = gauss_fit.params['center'].value
    ellip_bkgstd = gauss_fit.params['sigma'].value/np.sqrt(sky_ellip_aper.size)
    
    # Renctangular Apertures
    nboxes = 20 
    boxWidth = np.int64(bkgrad*np.pi/nboxes)
    boxWidth = 100 if boxWidth>100 else boxWidth
    boxWidth = 20 if boxWidth<20 else boxWidth
    halfBoxWidth = boxWidth//2
    nboxes =  np.int64(bkgrad*np.pi/boxWidth)

    rms = np.array([])
    medians = np.array([])
    stds = np.array([])
    cxs,cys=np.array([]),np.array([])
    newmask = ~(np.zeros(data.mask.shape, dtype=bool) + (smacoord>bkgrad)*(smacoord<=(scale*bkgrad+boxWidth))*~data.mask)
    good_coords = np.where(~newmask)

    if good_coords[0].size < nboxes*boxWidth:
        newmask = np.zeros(data.mask.shape, dtype=bool) + (smacoord<scale*bkgrad)*(smacoord>=(scale*bkgrad+4*boxWidth))
        good_coords = np.where(~newmask)
        raise Warning(f'Not enough unmasked pixels to measure noise in {nboxes} boxes.'
                      f' Only {good_coords[0].size} unmasked pixels available.')
        
    
    # Add edges to avoid border effects
    edge_mask = np.zeros(data.shape, dtype=bool) + True
    edge_mask[halfBoxWidth: -halfBoxWidth, halfBoxWidth: -halfBoxWidth] = False
    newmask[edge_mask] = True
    
    # Measure noise in nboxes
    for i in range(nboxes):
        idx = rng.choice(np.arange(len(good_coords[0])))
        ceny, cenx = good_coords[0][idx], good_coords[1][idx]
        box = data[ceny-halfBoxWidth:ceny+halfBoxWidth,
                   cenx-halfBoxWidth:cenx+halfBoxWidth]
        maskbox = data.mask[ceny-halfBoxWidth:ceny+halfBoxWidth,
                          cenx-halfBoxWidth:cenx+halfBoxWidth]
        rms = np.append(rms, np.sqrt(np.nansum(box[~maskbox]**2)/np.nansum(np.isfinite(box) * ~maskbox)))
        _,med,std = sigma_clipped_stats(box)
        medians = np.append(medians, med)
        stds = np.append(stds, std)
        newmask[ceny-halfBoxWidth:ceny+halfBoxWidth,
                   cenx-halfBoxWidth:cenx+halfBoxWidth] = True
        good_coords = np.where(~newmask)
        cxs = np.append(cxs,cenx)
        cys = np.append(cys,ceny)
        
    
    avRms = np.nanmean(rms)
    sbLim = np.nanstd(medians)
    rect_bkg= np.nanmedian(medians)
    rect_bkgstd = np.nanstd(medians)/np.sqrt(np.sum(~np.isnan(medians)))

    results = {'bkgrad' : bkgrad, 'avRms' : avRms, 
            'sbLim' : sbLim, 'ellip_bkg' : ellip_bkg, 'rect_bkg' : rect_bkg,
            'ellip_bkgstd' : ellip_bkgstd, 'rect_bkgstd' : rect_bkgstd,
            'nboxes':nboxes, 'boxWidth':boxWidth,
            'center': [center[0],center[1]], 'pa': pa, 'eps': eps, 'init': init,
            'cxs' : cxs.astype(np.int16).T, 'cys' : cys.astype(np.int16).T, 
            'rms' : rms.T, 'medians' : medians.T, 'stds' : stds.T
            }
    
    if plot != None:
        fig = plt.figure(figsize=(6,9))
        axim = plt.subplot2grid((6,2),(0,0),rowspan=3,colspan=2)
        axprof = plt.subplot2grid((6,2),(3,0),rowspan=2,colspan=2)
        axdprof = plt.subplot2grid((6,2),(5,0),colspan=2,sharex=axprof)
        
        vmax = 10**(0.4*(19.6-24))
        vmin = 10**(0.4*(19.6-29.5))
        norm = LogNorm(vmax=vmax, vmin=vmin)
        im = axim.imshow(data, norm=norm, cmap='nipy_spectral', origin='lower', interpolation='none')
        fig.colorbar(im, ax=axim)
        _ = rectangle_add_patches(np.array([results['cxs'],results['cys']]),results['boxWidth'],results['boxWidth'],axim,
                            linewidth=2, edgecolor='black', facecolor='none')
        axim.plot(*ellipse_points(results['center'], bkgrad, bkgrad*(1-results['eps']), results['pa'], num_points=300),c='k',ls='--',lw=2)
        axim.plot(*ellipse_points(results['center'], 1.1*bkgrad, 1.1*bkgrad*(1-results['eps']), results['pa'], num_points=300),c='k',ls='--',lw=2)
        
        axprof.scatter(x,y,color='gray',s=0.5,alpha=0.1)
        axprof.plot(sma,intensity,'ro')
        axprof.plot(sma,intsmooth,'g-')

        axprof.axhline(results['ellip_bkg'],c='k',ls='--', label='Elliptical')
        axprof.axhline(results['rect_bkg'],c='k',ls=':', label='Rectangular')
        axprof.plot(smacoord[results['cys'],results['cxs']],
            results['medians'],'s',ms=10, markeredgecolor='black', markerfacecolor='none',lw=6)
        axprof.axvline(results['bkgrad'],c='k',ls='-.', label='Radius')
        axprof.legend(loc='upper right',ncols=3, fontsize=12,frameon=False)

        axprof.set_yscale('log')
        axprof.set_ylim([results['ellip_bkg']-5*results['rect_bkgstd'], np.nanmax(intensity)])
        axprof.set_ylabel('I [Mjy]',fontsize=12)


        axim.set_xlim([results['center'][0]-bkgrad*1.5, results['center'][0]+bkgrad*1.5])
        axim.set_ylim([results['center'][1]-bkgrad*1.5, results['center'][1]+bkgrad*1.5])
        axprof.set_xlim([-0.03*bkgrad,1.5*bkgrad])
        axprof.set_ylim([results['rect_bkg']*0.03, np.nanmax(intensity)*1.1])

        axdprof.plot(sma,exp_crit,'r')
        axdprof.plot(sma,exp_crit_norm,'g')
        axdprof.axhline(med_exp,c='k',ls=':')
        axdprof.axhline(med_exp+5*std_exp,c='k',ls='--')
        axdprof.axvline(results['bkgrad'],c='k',ls='-.')
        axdprof.set_xlabel('semi-major axis [pixels]',fontsize=12)
        axdprof.set_ylabel(r'$|$I/dI/dr$|$',fontsize=12)
        axdprof.set_yscale('log')
        fig.tight_layout()
        fig.savefig(plot, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
    return results

def background_estimation(data, center, pa, eps, growth_rate = 1.03, out=None, verbose=False):
    '''
    Estimate the local background of a galaxy using elliptical apertures.
    
    It detects the maximum radius of the object containing signal with some 
    statistical tests [bkg_radius]. Then rectangular apertures photometry
    to estimate the background value and its uncertainty. 
    
    The apertures are created using the center, position angle, and ellipticity
    given. Then, the profile is meassure until certain radius. Once we reach 
    intensity values close to the mode + sigma, we increase the resolution
    (growth_rate=1.01), and compute the smooth derivative of the profile. 
    
    We stablish the value of the radius where the background is reach 
    this conditions are met:
        
        - The derivative change its sign for more than five times 
                This is a meassure of the fluctuation around dI = 0 
                when looking for a local assymptote.
        
        - Or when: I(r)/mode - 1 < epsilon [epsilon = 0.1]
                Safe parameter to avoid infinite loops.

    TODO: To faster the process, first compute distances to mode 
    +- sigma, and start in the closest or (q > 25 ¿?), and maybe 
    limit to (q < 75 ¿?).
    
    TODO: first guess background with sigma clipping distribution, 
    then do photometry with elliptical apertures increasing radius 
    and check evidence of convergence (or Kolmogorov–Smirnov test)
    and stop when next aperture is not signficante more similar. 
    Combine with current method.

    TODO: give option of sigma image to do statistical comparison of 
    value obtained with this, to see if the are of the same order, if not
    give warning.

    TODO: What is the best estimation of the uncertainty of the background
    meassure here?
        - Is it the std/np.sqrt(N_rect) 
        - Is it the rms or rms normalize by the square root?
    Do tests to see which is the best.

    Parameters
    ----------
        data : 2D numpy array
            Image of the galaxy.
        center : tuple
            Center of the galaxy in pixel coordinates.
        pa : float
            Position angle of the galaxy in degrees.
        eps : float
            Ellipticity of the galaxy. [1-b/a]
        growth_rate : float, optional
            Growth rate of the apertures. The default is 1.03.
        out : str, optional
            Path to save the background profile. The default is None.
        verbose : bool, optional
            Print the results. The default is False.
    
    Returns
    -------

        localsky : float
            Background value.
        localsky_std : float
            Background uncertainty.
        bkg_radius:  float
            Limiting radius where the background is reached.
    '''
    if not hasattr(data,'mask'): data = np.ma.array(data, mask=np.zeros_like(data))
    pa = pa*np.pi/180 # convert to radians
    
    # First guess of the mode value
    flatten = data[data.mask==False].flatten()
    flatten = flatten[np.isfinite(flatten)]
    mode, results = find_mode(flatten)
    std = results.values['sigma']

    rad = 10*np.ones(1)
    intensity = np.zeros(0)
    intensity_std = np.zeros(0)
    ellip_apertures = []
    previous_mask = np.zeros_like(data.mask)
    converge = False
    maxr,epsilon = np.NaN, 1e-1

    while not converge:

        if len(ellip_apertures) > 1:
            previous_mask = mask

        ellip_apertures.append(EllipticalAperture((center[0],center[1]), rad[-1], (1-eps)*rad[-1], pa))
        mask = ellip_apertures[-1].to_mask(method='center').to_image(data.shape)

        index = ut.where([data.mask==False,mask!=0,previous_mask==0])
        clipped = sigma_clip(data.data[index],sigma=3,maxiters=3)
        intensity= np.append(intensity, np.ma.median(clipped))
        intensity_std = np.append(intensity_std, np.nanstd(clipped)/np.sqrt(np.size(clipped)))
    
        if (intensity[-1] < mode + std) and np.isnan(maxr):

            growth_rate = 1.01
            index = intensity < mode + std
            dIdr = ut.derivative(rad[index],intensity[index])
            signs = np.sign(dIdr[1:]/dIdr[:-1]) == -1

            if np.sum(signs) > 5 or (np.abs(intensity[-1]/mode - 1) < epsilon):
                maxr = asymtotic_fit_radius(rad[index],dIdr)
                if verbose: print(f'maxr={maxr:.2f}; rad={rad[-1]:.2f}; intesity={intensity[-1]:.2f}')

        if rad[-1] > 1.3*maxr:
            converge = True
            if verbose: print(maxr,len(rad),len(intensity))
            break

        rad = np.append(rad, rad[-1]*growth_rate)


    dIdr = ut.derivative(rad,intensity)
    ddIdr2 = ut.derivative(rad,dIdr)

    index = intensity < mode + std
    skyradius1 = asymtotic_fit_radius(rad[index],dIdr[index])
    skyradius2 = asymtotic_fit_radius(rad[index],ddIdr2[index])

    skyradii = np.sort([skyradius1,skyradius2])
    aperfactor = np.nanmax([0.01,float(np.round((60 - np.diff(skyradii)) / (np.sum(skyradii)),3))])
    width = float(np.nanmax(np.diff(skyradii).tolist() + [60]))


    bkg_aperture = EllipticalAnnulus((center[0],center[1]),
                     (1-aperfactor)*skyradii[0], (1+aperfactor)*skyradii[1], 
                    (1-0.6*eps)*(1+aperfactor)*skyradii[1], None,
                    theta=pa)

    # Measure the background using the elliptical annulus
    mask_aper = bkg_aperture.to_mask(method='center').to_image(data.shape)
    mask_aper = np.ma.array(mask_aper,mask=1-mask_aper)
    aper_values = data*mask_aper
    aper_values = aper_values[np.where(~aper_values.mask)].flatten()
    localsky, gauss_fit = find_mode(aper_values)  
    aper_bkg = gauss_fit.params['center'].value
    aper_bkgstd = gauss_fit.params['sigma'].value/np.sqrt(aper_values.size)


    # Renctangular Apertures
    n_boxes = 20 
    width_boxes =  np.int64(0.9*(np.mean(skyradii)*np.pi/n_boxes))
    if width_boxes>100:
        width_boxes=100
    elif width_boxes<10:
        width_boxes=20
        n_boxes =  np.int64(0.9*(np.mean(skyradii)*np.pi/width_boxes))

    rect = random_rectangular_boxes(center,-pa, np.mean(skyradii), 0.6*eps, n=n_boxes, wbox=width_boxes)

    res_stats = []
    for r in rect:
        mask_r =r.to_mask(method='center').to_image(data.shape)
        aper_val = data[(~data.mask) * (mask_r==1)]
        mean,med,std = sigma_clipped_stats(aper_val.flatten())
        res_stats.append(med)

    # Use rectangular localsky estimate
    localsky = np.nanmedian(res_stats)
    localsky_std = np.nanstd(res_stats)/np.sqrt(np.sum(~np.isnan(res_stats)))

    # Plotting
    if out is not None: 
        plt.rcParams.update({"text.usetex": False})
        fig = plt.figure(figsize=(12,8))
        ax1 = plt.subplot2grid((4,3),(1,0))
        ax2 = plt.subplot2grid((4,3),(2,0),sharex=ax1)
        ax3 = plt.subplot2grid((4,3),(3,0),sharex=ax1)
        ax4 = plt.subplot2grid((4,3),(1,1),rowspan=3,colspan=2)
        axtext = plt.subplot2grid((4,3),(0,0),colspan=2)
        axtext.axis('off')
        
        fontsize = 12
        ax1.plot(rad,intensity,'.')
        ax1.set_ylabel('Intensity (ADUs)',fontsize=fontsize)
        ax1.axhline(mode,ls='--',c='k',label='Mode')
        ax1.axhline(mode+std,ls=':',c='k',label='Mode $\pm \sigma$')
        ax1.axhline(aper_bkg, ls='-.', c='magenta', label='Aperture')
        ax1.axhline(localsky, ls='-.', c='green', label='Boxes')
        ax1.set_ylim([mode-std*0.3,mode+std*1.4])
        arglim = np.nanargmin(np.abs(intensity-mode-std))
        ax1.set_xlim([rad[arglim],1.1*np.nanmax(np.append(rad[-1],skyradii))])
        ax1.legend(fontsize=10,loc='upper left',ncol=2, frameon=False)

        ax2.plot(rad,dIdr,'.')
        ax2.set_ylabel('$dI/dr$',fontsize=fontsize)
        ax2.axhline(0,ls='--',c='k')
        ax2.axvline(skyradius2,ls='--',c='r',label='Sign change')
        ax2.set_ylim([dIdr[arglim],-dIdr[arglim]/10])

        ax3.plot(rad,ddIdr2,'.')
        ax3.set_ylabel('$d^2I/dr^2$',fontsize=fontsize)
        ax3.set_xlabel('Radius (pixels)',fontsize=fontsize)
        ax3.axhline(0,ls='--',c='k')
        ax3.axvline(skyradius1,ls='--',c='r',label='Sign change')
        ax3.set_ylim([-ddIdr2[arglim]/10,ddIdr2[arglim]])

        for ax in [ax1,ax2,ax3]:
            ax.axvline(bkg_aperture.a_in,ls='-.',c='magenta',label='Sky annulus')
            ax.axvline(bkg_aperture.a_out,ls='-.',c='magenta')       
            ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0)) 

        show(data-localsky, vmin=gauss_fit.params['sigma'].value, ax=ax4)
        width=1.2*skyradii[1]
        ax4.set_xlim([center[0]-width,center[0]+width])
        ax4.set_ylim([center[1]-width,center[1]+width])
        bkg_aperture.plot(axes=ax4,color='black',lw=1)
        

        text = f'''\
Background estimation using elliptical apertures of \
{os.path.basename(os.path.splitext(out)[0])} 
_____________________________________________________________________________

center = ({center[0]:.2f} , {center[1]:.2f})     \
PA = {pa*180/np.pi:.2f} degrees [x to y]     \
Ellipticity = {eps:.2f}

Background Meassured in an elliptical annulus of width {skyradii[1]-skyradii[0]:.2f} pixels
Mode  = {aper_bkg:.3e} +- {gauss_fit.params['center'].stderr:.3e}      \
Sigma = {gauss_fit.params['sigma'].value:.3e} +- {gauss_fit.params['sigma'].stderr:.3e}
Background = {aper_bkg:.3e} +- {aper_bkgstd:.3e}

Background Meassured in {n_boxes:d} boxes of width {width_boxes:d} pixel \
at a distance of {float(np.nanmax([rad[-1],maxr])):.2f} pixels from the center
Background = {localsky:.3e} +- {localsky_std:.3e}
'''

        axtext.text(0.05, 1.05, text,
            ha='left', va='top', transform=axtext.transAxes, fontsize=10)
        rect.plot(axes=ax4,color='black')
        fig.subplots_adjust(hspace=0.4)
        # ax4.text(0.02, 1.05, os.path.basename(os.path.splitext(out)[0]), ha='left',
        #         va='bottom', transform=ax4.transAxes, fontweight='bold',fontsize='large')
        fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.rcParams.update({"text.usetex": True})


    return aper_bkg, localsky_std, float(np.nanmax([rad[-1],maxr]))


def autoprof_isophote_photometry(data, center, pa_init, eps_init, growth=0.05,
                        fit_limit=2, smooth=1, background = 0, bkgstd = None, psf=1):
    ''' Autoprof wraper of Isophote_Fit_FFT_Robust to use within astropipe. 
    Given an image and its morphological parameters, fits ellitpical regions 
    to isophotes and returns the radial profile of the galaxy.
    
    PARAMETERS
    ----------
        data : array_like
            2D array with the image data.
        center : tuple
            (x,y) coordinates of the center of the galaxy [pixels].
        pa_init : float
            Initial position angle of the galaxy [degrees].
        eps_init : float
            Initial ellipticity of the galaxy.
        growth : float, optional
            Growth rate of the isophotes. The default is 0.05.
        fit_limit : float, optional
            Limit of the fit. The default is 2. (lower values
            will fit more isophotes). "ap_fit_limit" in the options.
        smooth : float, optional
            Smoothing factor. The default is 1. Larger values will
            fit smoother isophotes. "ap_regularize_scale" in the options.
        background : float, optional
            Background level. The default is None.
        bkgstd : float, optional
            Background noise level. The default is None.
        psf : float, optional
            Point spread function full width at half maximum. The default is 1.
    
    RETURNS
    -------
        profile : astropipe.profile.Profile
            Object with the radial profile of the galaxy.
    '''
    
    # AutoProf required options 
    options = {
    "ap_scale": growth, "ap_fit_limit": fit_limit, "ap_regularize_scale": smooth,
    "ap_isofit_robustclip": 0.15, "ap_isofit_losscoefs": (2,),
    "ap_isofit_superellipse": False, "ap_isofit_fitcoefs": (2, 4),  
    "ap_isofit_fitcoefs_FFTinit": True, "ap_isofit_perturbscale_ellip": 0.03,
    "ap_isofit_perturbscale_pa": 0.06, "ap_isofit_iterlimitmax": 300,
    "ap_isofit_iterlimitmin": 0, "ap_isofit_iterstopnochange": 3,
    "ap_doplot": False,  "ap_name": "GalaxyIsophoteFit"}
    
    mask = data.mask if hasattr(data,'mask') else np.zeros_like(data)
    if bkgstd is None: _,_,bkgstd = sigma_clipped_stats(data, mask=mask)

    results = { "background": 0, "background noise": bkgstd,
    "psf fwhm": psf, "init ellip": eps_init,
    "center": {'x':center[0],'y':center[1]}, 
    "init pa": pa_init * np.pi / 180, "mask": mask }    


    _, results = Isophote_Fit_FFT_Robust(data, results, options)

    keys = ['fit ellip', 'fit pa', 'fit R',
            'fit ellip_err', 'fit pa_err', 'auxfile fitlimit',
            'fit Fmodes', 'fit Fmode A2', 'fit Fmode Phi2', 
            'fit Fmode A4', 'fit Fmode Phi4']

    keys_to_remove = ['auxfile fitlimit', 'fit Fmodes']
    for key in keys_to_remove:
        if key in results:
            del results[key]

    tab = Table(results)
    tab['x'] = center[0]*np.ones_like(tab['fit R'])
    tab['y'] = center[1]*np.ones_like(tab['fit R'])
    tab.meta = options
    return tab

def res_sum_squares(dmdr, cog, slope, abcissa):

    y2 = abcissa+slope*dmdr    
    rms = np.mean(np.sqrt((cog-y2)**2))

    return rms, y2 

def asymtotic_fit_radius(x,y):
    '''
    Find the asymptotic radius of a profile
    fitting a line and finding the intersection with the y axis
    '''
    xx = np.array(x)
    yy = np.array(y)
    want = xx == xx
    for i in range(3):
        xx = xx[want]
        yy = yy[want]
        fit = stats.linregress(xx, yy)
        rms, y2 = res_sum_squares(xx, yy, fit[0], fit[1])  # This was a custom function I wrote
        want = np.abs(yy - y2) <= 3*rms
    
    return -fit[1]/fit[0]

def measureImageNoise(maskedImageArray,
                      galX,
                      galY,
                      galRad, galPa, galEps,
                      halfBoxWidth=10,
                      nboxes=50,
                      seed=None,
                      plot=False):
    '''
    Outputs metrics for the background noise and flux.

    Parameters
    ----------
    maskedImageArray : numpy.ma.core.MaskedArray
        Image with interloping sources masked
    galX : float
        Central x-coordinate of target galaxy
    galY : float
        Central y-coordinate of target galaxy
    galRad : int
        Radius out to which to apply circular mask to target galaxy [pixel]
    galPa : float
        Position angle of target galaxy [degrees]. X to Y direction. Anti-clockwise.
    galEps : float
        Ellipticity of target galaxy [1-b/a]
    halfBoxWidth : int, optional
        Half the desired width of the boxes used to calculate the noise.
        The default is 10.
    seed : int, optional
        A seed for the random generator, for testing purposes

    Returns
    -------
    avRms : float
        The average root mean square among counts in nboxes randomly
        distributed boxes, ignoring masked pixels
    sbLim : float
        The standard deviation of the median values within nboxes randomly
        distributed boxes, ignoring masked pixels
    sky : float
        Median of the median values within nboxes randomly distributed boxes,
        ignoring masked pixels
    skystd : float
        Error on de median value sky. std/sqrt(nboxes).
    '''
    rng = np.random.default_rng(seed)
    # Need to avoiding actually altering the image and mask used
    if type(maskedImageArray) == np.ma.core.MaskedArray:
        data = np.zeros(maskedImageArray.shape) + maskedImageArray.data
        mask = np.zeros(maskedImageArray.shape, dtype=bool)\
            + maskedImageArray.mask
        maskedImageArray = np.ma.masked_array(data, mask=mask)
    else:
        maskedImageArray = np.zeros(maskedImageArray.shape) + maskedImageArray

    # Boundaries: avoid masked interlopers, the target galaxy, and image edges
    edge_mask = np.zeros(maskedImageArray.data.shape, dtype=bool) + True
    edge_mask[halfBoxWidth: -halfBoxWidth, halfBoxWidth: -halfBoxWidth] = False
    # width = np.sqrt(np.pi*galRad**2 - 2*(2*halfBoxWidth)**2) - galRad
    nExtra = np.pi*galRad / (halfBoxWidth*2)**2
    width = np.sqrt(2*(nboxes+nExtra)*(2*halfBoxWidth)**2/np.pi)
    ellipannulus = EllipticalAnnulus((galX,galY), 
                        galRad, galRad+width, 
                        (1-0.6*galEps)*(galRad+width),
                        theta=galPa*np.pi/180)
    regionOfInterest = ellipannulus.to_mask(method='center').to_image(data.shape).astype(bool)
    mask = np.array(maskedImageArray.mask + ~regionOfInterest + edge_mask).astype(bool)
    
    # Measure stats in boxes
    good_coords = np.where(~mask)
    if good_coords[0].size < nboxes*halfBoxWidth*2:
        raise Warning(f'Not enough unmasked pixels to measure noise in {nboxes} boxes.'
                      f' Only {good_coords[0].size} unmasked pixels available.')
    
    if plot:
        ax=show(np.ma.masked_array(data, mask=mask))

    rms = np.array([])
    medians = np.array([])
    stds = np.array([])
    newmask = np.zeros(mask.shape, dtype=bool) + mask
    for i in range(nboxes):
        idx = rng.choice(np.arange(len(good_coords[0])))
        ceny, cenx = good_coords[0][idx], good_coords[1][idx]
        box = data[ceny-halfBoxWidth:ceny+halfBoxWidth,
                   cenx-halfBoxWidth:cenx+halfBoxWidth]
        maskbox = mask[ceny-halfBoxWidth:ceny+halfBoxWidth,
                          cenx-halfBoxWidth:cenx+halfBoxWidth]
        rms = np.append(rms, np.sqrt(np.nansum(box[~maskbox]**2)/np.sum(~np.isnan(box) * ~maskbox)))
        _,med,std = sigma_clipped_stats(box)
        medians = np.append(medians, med)
        stds = np.append(stds, std)
        newmask[ceny-halfBoxWidth:ceny+halfBoxWidth,
                   cenx-halfBoxWidth:cenx+halfBoxWidth] = True
        good_coords = np.where(~newmask)
        if plot:
            aperture = RectangularAperture((cenx, ceny), w=halfBoxWidth, h=halfBoxWidth)
            aperture.plot(ax, color='black', lw=2)

    avRms = np.nanmean(rms)
    sbLim = np.nanstd(medians)
    sky = np.nanmedian(medians)
    skystd = np.nanstd(medians)/np.sqrt(np.sum(~np.isnan(medians)))

    

    return avRms, sbLim, sky, skystd


# Find radius where an asintote starts in a profile
def find_radius_asintote(x,y):
    # Find the first point where the slope of the line is 0
    # This is the point where the asintote starts
    # x,y are the profile
    # Returns the radius where the asintote starts

    # Find the slope of the line between each point
    slope = ut.derivative(x,y)

    # Find the first point where the slope is 0
    # This is the point where the asintote starts
    # This is the point where the slope changes sign
    sign_change = np.where(np.diff(np.sign(slope)))[0][0]

    # Return the radius where the asintote starts
    return x[sign_change]
    
def find_mode(data):
    mean, median, std = sigma_clipped_stats(data,sigma=3)
    hist, bin_edges = np.histogram(data,bins=1000,
                         range=[median - 5*std, median + 5*std])
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    model = GaussianModel()
    params = model.guess(hist,x=bin_centers)
    result = model.fit(hist,params,x=bin_centers)    

    return  result.values['center'], result



def aaron_break_finder(rad,mu,min=21,max=31,n=4,p=5):
    """
    Finds the disk breaks in the surface brightness profile as 
    seen in Watkins et al. (2019)
    """
    index = ut.where([mu>min,mu<max,np.isfinite(mu)])[0]
    der = ut.derivative(rad[index],mu[index],n=n)
    der = median_filter(der,size=int(p*len(mu)/100))
    cum_sum = np.cumsum(der-np.mean(der))
    maximum = rad[index][argrelextrema(cum_sum, np.greater)[0]]
    minimum = rad[index][argrelextrema(cum_sum, np.less)[0]]
    critic =  np.sort(np.append(maximum,minimum))
    arg = []
    for point in critic:
        arg.append(ut.closest(rad,point)+1)
    return arg



def find_slope(rad, mu, w=2, smFac=0.1):
    '''
    Derive slope of surface brightness profile vs. radius using adjacent w data
    points.  Returns this and a median-smoothed version of it.

    Requires:
     - rad (numpy.array): radius array
     - mu (numpy.array): surface brightness array, in mags/arcsec^2
     - w (int): width of array segments used to calculate local slope
     - smFac (float): fraction of array used as smoothing kernel size

    Returns:
     - inrad (numpy.array): clipped radius array of proper size
     - smooth_h (numpy.array): median filtered local scale lengths
     - h (numpy.array): unfiltered local scale lengths
    '''
    inrad = rad[w:-w]
    m = np.array([])
    for r in range(len(inrad)):
        idx = np.where(rad == inrad[r])[0][0]
        fit_rads = rad[idx-w:idx+w]
        fit_flux = mu[idx-w:idx+w]
        f = np.polyfit(fit_rads, fit_flux, 1)
        m = np.append(m, f[0])
        
    kernel = round(smFac*len(m), 0)
    if kernel % 2 == 0:
        kernel += 1
    smooth_m = medfilt(m, int(kernel))
    h = 1.0857/m  # Scale length
    smooth_h = 1.0857/smooth_m
    
    return inrad, smooth_h, h


def cusum_wat(x):
    '''
    Cumulative sum of the difference from the mean.
    The minimum or maximum is the break point.

    Requires:
     - x (numpy.array): the array for which you want the CUSUM

    Returns:
     - s (numpy.array): the CUSUM of x
    '''
    xbar = np.mean(x)
    s = np.zeros(len(x))
    for i in range(len(x)):
        if i == 0:
            s[i] = 0
        else:
            s[i] = s[i-1] + (x[i]-xbar)
            
    return s

def cusum(x):
    '''
    Cumulative sum of the difference from the mean.
    The minimum or maximum is the break point.

    Requires:
     - x (numpy.array): the array for which you want the CUSUM

    Returns:
     - c (numpy.array): the CUSUM of x
    '''
    c = np.cumsum(x-np.mean(x))
    c[0] = 0
    return c

def change_point(inrad, h, N=1e5):
    '''
    Finding the change point of the radial slope profile using the method illustrated here:
    http://www.variation.com/cpa/tech/changepoint.html

    Requires:
     - inrad (numpy.array): clipped radius array output from find_slope()
     - h (numpy.array): a local slope array output from find_slope()
     - N (int): number of bootstrapping iterations for confidence estimate

    Returns:
     - i1 (int): index in inrad of change point
     - s (numpy.array): cumulative sum profile
     - conf (float): percentage of bootstrapping runs in which a break was found

    NOTE: can use either h or smooth_h; whichever you feel does the better job
    NOTE2: for confidence estimate, consider also applying the
    criterion that the break must occur in the same general place in
    the array, to enhance robustness.  Not currently implemented.
    '''
    s = cusum(h)
    # Amplitude of the break
    sdiff = np.max(s) - np.min(s)
    
    # Confidence estimate by bootstrapping
    # print('Bootstrapping...')
    bs_sdiff = np.array([])
    for i in range(np.int64(N)):
        randi = np.random.randint(0, len(h), len(h))
        randh = h[randi]
        rands = cusum(randh)
        randsdiff = np.max(rands) - np.min(rands)
        bs_sdiff = np.append(bs_sdiff, randsdiff)
    good = bs_sdiff < sdiff
    conf = len(good[good])/len(good)
    
    # Location estimate one: maximum in s
    if abs(np.max(s)) > abs(np.min(s)):
        i1 = np.where(s == np.max(s))[0][0]
    else:
        i1 = np.where(s == np.min(s))[0][0]
    
    return i1, s, conf


def find_all_breaks(inrad, h, rin, rout):
    '''
    Tailored to find a maximum of three breaks across the disk.  Will
    require manual adjustment to find more than this.

    Requires:
     - inrad (numpy.array): clipped radius array output from find_slope()
     - h (numpy.array): a local slope array output from find_slope()
     - rin (float): inner radius boundary for searching for breaks
     - rout (float): outer radius boundary for searching for breaks

    Returns:
     - rbr1, rbr2, rbr3 (float): the three break radii the routine
       found, in the same units as inrad array

    NOTE: if a break wasn't found to be significant, this returns -999
    '''
    want = (inrad >= rin) & (inrad <= rout) & (~np.isinf(h)) & (~np.isnan(h))
    goodrad = inrad[want]
    goodh = h[want]
    
    # First run, find if there's a global break
    ig, sg, confg = change_point(goodrad, goodh)
    # Check confidence; if less than 95% of bootstrap runs found a
    # break, the global break isn't significant, so it doesn't bother to
    # look for any beyond that.
    if confg < 0.95:
        print('No breaks found')
        return -999, -999, -999
    else:
        rbr2 = goodrad[ig]
        want_in = (goodrad < rbr2)
        want_out = (goodrad > rbr2)
        i_in, s_in, conf_in = change_point(goodrad[want_in], goodh[want_in])
        i_out, s_out, conf_out = change_point(goodrad[want_out], goodh[want_out])
        # print('Initial guess: ',goodrad[want_in][i_in], goodrad[ig], goodrad[want_out][i_out])
        # Same here: always using 95% confidence threshold (p-value = 0.05)
        if conf_in < 0.95:
            rbr1 = -999
        else:
            rbr1 = goodrad[want_in][i_in]
        if conf_out < 0.95:
            rbr3 = -999
        else:
            rbr3 = goodrad[want_out][i_out]
            
        # print(conf_in, confg, conf_out)
        return rbr1, rbr2, rbr3

def break_estimation(radius, mu, rms, skyrms=0, 
        rin=7,rout=150, npx = 4, zp=22.5 , pixel_scale=0.333):

    '''
    Estimate three different break in a surface brightness radial profile.
    The consistance of the breaks are also estimated by adding noise to the profile. 
    Requires:
     - radius (numpy.array): radius array in arcsec
     - mu (numpy.array): mu array in counts
     - std (numpy.array): std array in counts
     - skyrms (float): sky rms level in counts
     - npx (int): number of pixels in the image
     - zp (float): zeropoint of the image
     - pixel_scale (float): pixel scale in arcsec/pixel
    Returns:
     - rbr (float): break radius in arcsec
     - rbr_err (float): error on break radius in arcsec
    '''

    sig = np.sqrt((np.sqrt(skyrms**2)/np.sqrt(npx))**2 + rms**2)
    intensity = (pixel_scale**2) * np.power(10,(zp - mu)/2.5)
    inrad, smooth_h, h = find_slope(radius, mu)

    mu_up = -2.5*np.log10(intensity+sig) + zp
    mu_up[np.isnan(mu_up)] = -999
    inrad_up, smooth_h_up, h_up = find_slope(radius, mu_up)

    mu_down = -2.5*np.log10(intensity+sig) + zp
    mu_down[np.isnan(mu_down)] = -999
    inrad_down, smooth_h_down, h_down = find_slope(radius, mu_down)

    
    # Unaltered profile
    rbr1, rbr2, rbr3 = find_all_breaks(inrad, smooth_h, rin, rout)
    # First sky perturbation
    rbr1_up, rbr2_up, rbr3_up = find_all_breaks(inrad_up, smooth_h_up, rin, rout)
    # Second sky perturbation
    rbr1_down, rbr2_down, rbr3_down = find_all_breaks(inrad_down, smooth_h_down, rin, rout)

    rbr = np.array([rbr1,rbr2,rbr3,
                    rbr1_up,rbr2_up,rbr3_up,
                    rbr1_down,rbr2_down,rbr3_down])
    if any(rbr>0):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(rbr[rbr>0].reshape(-1,1))
        centers = kmeans.cluster_centers_
    else:
        centers = [-999,-999,-999]

    return centers

def create_ellipse_meshgrid(eps, pa, center, grid_shape):
    """
    Create meshgrid of semimajor axis following an elliptical shape.
    
    Parameters:
        eps (float): Ellipticity of the ellipse (0 <= e < 1).
        pa (float): Position angle of the ellipse in degrees (counterclockwise from x-axis).
        center (tuple): Center of the ellipse (cx, cy).
        grid_shape (tuple): Shape of the meshgrid (height, width).
    
    Returns:
        x_coords (2D array): X-coordinates of the meshgrid.
        y_coords (2D array): Y-coordinates of the meshgrid.
        mask (2D array): Boolean mask where `True` indicates points inside the ellipse.
    """    
    # Center of the ellipse
    cx, cy = center

    # Create a regular meshgrid
    height, width = grid_shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Shift the grid to the ellipse center
    x_shifted = x - cx
    y_shifted = y - cy

    # Convert position angle to radians
    pa_rad = np.radians(pa)

    # Rotate the coordinates by the position angle
    x_rot = x_shifted * np.cos(pa_rad) + y_shifted * np.sin(pa_rad)
    y_rot = -x_shifted * np.sin(pa_rad) + y_shifted * np.cos(pa_rad)

    # Ellipse equation
    sma = np.sqrt(x_rot**2 + y_rot**2 / ((1-eps)**2))

    return sma


def create_matrix_by_bins(x, y, bins):
    '''
    Create a matrix z where each row contains values of y corresponding to ranges of x given by bins.

    Parameters
    ----------
        x : array
            x-axis values.
        y : array
            y-axis values.
        bins : array
            Bin edges.

    Returns
    -------
        z : 2D array
            Matrix where each row contains the values of y corresponding to each bin of x.
            Rows with no values in the bin will be padded with NaN.
    '''
    # Ensure inputs are NumPy arrays
    x = np.array(x)
    y = np.array(y)
    bins = np.array(bins)

    # Digitize x values into bin indices
    bin_indices = np.digitize(x, bins) - 1  # Get bin indices (0-indexed)

    # Create a mask for valid bin indices
    valid_mask = (bin_indices >= 0) & (bin_indices < len(bins) - 1)

    # Filter out invalid values
    x = x[valid_mask]
    y = y[valid_mask]
    bin_indices = bin_indices[valid_mask]

    # Find the maximum number of elements in any bin
    counts = np.bincount(bin_indices, minlength=len(bins) - 1)
    max_count = counts.max()

    # Sort arrays 
    sorting = np.argsort(bin_indices)
    x = x[sorting]
    y = y[sorting]
    bin_indices = bin_indices[sorting]
    col_indices = np.concatenate([np.arange(np.sum(bin_indices==b)) for b in np.unique(bin_indices) ])

    # Create the output matrix
    z = np.full((len(bins) - 1, np.nanmax(col_indices)+1), np.nan)  # Initialize with NaN

    # Fill the rows of the matrix with y values for each bin
    z[bin_indices, col_indices] = y

    return z


def elliptical_profile_fast(data, rad, center, pa, eps):

    x = create_ellipse_meshgrid(eps, pa, center, data.shape).flatten()[~data.mask.flatten()]
    y = data.flatten()[~data.mask.flatten()]

    z = create_matrix_by_bins(x,y,rad)
    _,intensity,intensity_err = sigma_clipped_stats(z,axis=1)
    
    return intensity,intensity_err