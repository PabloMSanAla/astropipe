import numpy as np
from matplotlib import pyplot as plt
from astropy.stats import SigmaClip
from sklearn.utils import resample
from photutils.centroids import centroid_2dg
from astropipe.profile import elliptical_radial_profile


'''Create Star Mask
    1) Create Star Mask
        1.1) Filter Star Mask (Crow) ¿?
        1.2) Filter Bright (tails) and Faint (center)
    2) Compute Normalization value
        2.1) Fit star to Moffat 
        2.2) Fix center
        2.3) Compute normalization term 35% of light
        2.4) Compute sky ring from 80-90% of light
        2.5) Update catalog with these values
    3) Create PSF
'''

def radial_average2D(array,width=1,method='sigma-clipping'):
    ''' Radial average of a numpy array. The center
    of the radial average is assume to be the center of
    the image. 
    Input:
        :array: ndarray to radial average.
        :width: width of the radial bins. Pixels inside this bin
            would be average. 
        :method: ['mean','median','sigma-clipping']
    Output:
        :radial: ndarray of the radial average result 
    '''
    if method=='mean': aggregation = np.mean
    elif method=='median': aggregation = np.median
    elif method=='sigma-clipping': aggregation = SigmaClip(sigma=2., maxiters=None)
    else: raise ValueError('method not recognized')
    
    x = np.arange(0,array.shape[0])
    y = np.arange(0,array.shape[1])
    X,Y = np.meshgrid(x,y)
    Z = np.sqrt((X-array.shape[0]/2)**2 + (Y-array.shape[1]/2)**2)

    radial = np.zeros_like(array)
    i = width
    while i < Z.max():
        index = np.where((Z>i-width) & (Z<i+width))
        radial[index] = aggregation(array[index])
        if method=='sigma-clipping': radial[index] = np.mean(radial[index])
        i += width  
    return radial


# funtion that radially average an image
def radial_average1D(array):
    # create a grid of the same size as the image
    y, x = np.indices(array.shape)
    # compute the center of the image
    center = np.array([(x.max() - x.min()) / 2.0, (x.max() - x.min()) / 2.0])
    # compute the radius of each pixel from the center
    r = np.hypot(x - center[0], y - center[1])
    # compute the average value of all pixels with the same radius
    tbin = np.bincount(r.astype(int).ravel(), array.ravel())
    nr = np.bincount(r.astype(int).ravel())
    radialprofile = tbin / nr
    return radialprofile


def create_stars_cutouts(image, star_positions, cutout_size=61):
    """
    Create cutouts of stars from the image based on given positions.
    
    Parameters
    ----------
    image : 2D ndarray
        The input image data.
    star_positions : list of tuples
        List of (x, y) pixel coordinates for star positions.
    cutout_size : int
        Size of the square cutout (in pixels, should be odd).
    
    Returns
    -------
    cutouts : list of 2D ndarrays
        List of star cutout images.
    """
    from astropy.nddata import Cutout2D
    
    # Ensure cutout_size is odd
    if cutout_size % 2 == 0:
        cutout_size += 1
        print(f"Cutout size adjusted to {cutout_size} to ensure it is odd.")
    
    cutouts = []
    for (x, y) in star_positions:
        try:
            cutout = Cutout2D(image, (x, y), cutout_size)
            cutouts.append(cutout.data)
        except Exception as e:
            print(f"Warning: Could not create cutout for star at ({x}, {y}): {e}")
    return cutouts

def plot_cutout(cutouts, zp=0, pixel_scale=1, **kwargs):
    '''
    Helper function to plot star cutouts with magnitude scaling.

    Parameters
    ----------
    cutouts : list of 2D ndarrays
        List of star cutout images.
    zp : float
        Zero point for magnitude calculation.
    pixel_scale : float
        Pixel scale in arcsec/pixel for magnitude calculation. 
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plotted star cutouts.
    '''

    ncut = len(cutouts)
    ncols = int(np.ceil(np.sqrt(ncut)))
    nrows = int(np.ceil(ncut / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10*ncols/nrows, 10))
    axes = axes.flatten()
    for i, cutout in enumerate(cutouts):
        mag = zp - 2.5*np.log10(cutout) + 5*np.log10(pixel_scale)
        axes[i].imshow(mag, origin='lower', **kwargs)
        axes[i].set_title(f'Star {i}', fontsize=12)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    for k in range(i + 1, len(axes)):
        axes[k].axis('off')
    fig.suptitle('Star Cutouts', fontsize=14)
    fig.tight_layout()
    return fig


def resample(data, factor=4):
    """
    Resample cutout by a given factor using skimage resize with bilinear interpolation.
    
    Parameters
    ----------
    data : ndarray
        Original cutout data
    factor : int
        Resampling factor (e.g., 4 for 4x upsampling)
    
    Returns
    -------
    resampled : ndarray
        Resampled star data with shape (factor*size, factor*size)
    """
    from skimage.transform import resize
    
    # Get original size
    ny, nx = data.shape
    output_shape = (ny * factor, nx * factor)
    
    # Resize using bilinear interpolation (order=1)
    resampled = resize(data, output_shape, order=1, mode='constant', cval=0, preserve_range=True)
    
    return resampled

def recenter_stars(stars_data_list, centroid_func=centroid_2dg):
    """
    Recenter star cutouts to their center.
    
    Parameters
    ----------
    stars_data_list : list of ndarray
        List of resampled star cutout data
    centroid_func : callable
        Function to compute centroid (default: centroid_2dg)
    
    Returns
    -------
    recentered : list of ndarray
        Recentered star data
    offsets : ndarray
        Array of (dx, dy) offsets for each star
    """
    from scipy import ndimage
    
    recentered = []
    offsets = []
    
    for i, star_data in enumerate(stars_data_list):
        try:
            # Compute centroid
            yc, xc = centroid_func(star_data)
            
            # Get cutout size
            shape = star_data.shape
            center = (shape[0] - 1) / 2.0
            
            # Calculate offset in resampled coordinates
            dy = center - yc
            dx = center - xc
            offsets.append((dx, dy))
            
            # Shift the star to center it
            shifted_data = ndimage.shift(star_data, (dy, dx), order=3, mode='constant', cval=0)
            recentered.append(shifted_data)
        except Exception as e:
            print(f"Warning: Could not recenter resampled star {i}: {e}")
            recentered.append(star_data)
            offsets.append((0, 0))
    
    return recentered, np.array(offsets)

def normalize_star_cutouts(stars_data_list, methods, rmin, rmax, niter=3, reference_psf=None, plot=False):
    """
    Normalize star cutouts using radial profile matching.
    
    This function normalizes star cutouts by computing their radial brightness profiles
    and adjusting their flux to match a reference profile. Different normalization 
    methods can be applied to saturated and unsaturated stars.
    
    Parameters
    ----------
    stars_data_list : list of ndarray
        List of star cutout data arrays to normalize.
    methods : ndarray of int
        Normalization method for each star:
        - 0: Normalize unsaturated stars to their peak brightness
        - 1+: Normalize saturated stars to median difference in annulus defined by 
              rmin[method-1] and rmax[method-1]
    rmin : list of float
        Minimum radii (pixels) for each annulus region.
    rmax : list of float
        Maximum radii (pixels) for each annulus region.
    niter : int, optional
        Number of iterations for normalization convergence (default: 3).
    reference_psf : ndarray, optional
        Reference PSF for profile comparison. If None, uses first unsaturated star 
        (method==0) as reference (default: None).
    plot : bool, optional
        If True, creates visualization of radial profiles before and after normalization 
        (default: False).
    
    Returns
    -------
    normalized : list of ndarray
        List of normalized star cutout data.
    
    Examples
    --------
    >>> # Normalize 3 stars: first unsaturated (method=0), second and third saturated (method=1 and method=2, respectively)
    >>> method = np.array([0, 1, 2])
    >>> rmin = [40, 10]  # annulus radii
    >>> rmax = [50, 20]
    >>> normalized = normalize_star_cutouts(
    ...     cutouts, method, rmin=rmin, rmax=rmax, niter=3, plot=True
    ... )
    
    Notes
    -----
    - Unsaturated stars (method=0) normalize to their peak brightness
    - Saturated stars (method>0) normalize to median profile difference in the 
      specified annulus to avoid bright pixels
    - Profile is updated iteratively to improve convergence
    """
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*converting a masked element to nan.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered in log10.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*All-NaN slice encountered.*')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*Degrees of freedom <= 0.*')

    # Create a reference profile from the provided PSF or the first unsaturated star
    if reference_psf is  None:
        reference_psf = np.array(stars_data_list)[methods==0][0]  # Use the first unsaturated star as reference if no PSF provided
    
    # Measure the profile of the reference PSF
    eps, pa = 0.01, 0.01  # Ellipticity and position angle for radial profile (can be adjusted if needed)
    ref_profile = elliptical_radial_profile(reference_psf, reference_psf.shape[0]//2, (reference_psf.shape[0]//2, reference_psf.shape[1]//2), eps=eps, pa=pa)
    ref_profile.brightness()
    ref_profile.mu -= ref_profile.mu[0]  # Normalize reference profile to its peak

    normalized = []
    mag_differences = []
    profiles = []

    # Index for annulus normalization for each pair of methods and rmin,rmax
    index_norm = np.zeros((len(ref_profile.rad),len(rmin)), dtype=bool)
    for i in range(len(rmin)):
        index_norm[:, i] = (ref_profile.rad > rmin[i]) & (ref_profile.rad < rmax[i])

    # Prepare plot if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(stars_data_list)))

    # Start the iterative normalization process
    for k in range(niter):
        profiles = []
        for i,star in enumerate(stars_data_list):
            # Measure the profile of the current star
            profile = elliptical_radial_profile(star, star.shape[0]//2, (star.shape[0]//2, star.shape[1]//2), eps=eps, pa=pa)
            profile.brightness()

            # Current normalization method for this star
            if methods[i] == 0:
                magdiff = profile.mu[0]
            elif methods[i] > 0: 
                magdiff = np.nanmedian(profile.mu[index_norm[:, methods[i]-1]]-ref_profile.mu[index_norm[:, methods[i]-1]])
            
            # Store to update the reference profile using the median of all profiles
            profiles.append(profile.mu - magdiff)

            # Last iteration: Save normalized star and plots if requested
            if k == niter - 1: 
                mag_differences.append(magdiff)
                normalized_star = star * 10**(0.4*magdiff)
                normalized.append(normalized_star)
    
                if plot:
                    ax1.plot(profile.rad, profile.mu, color=colors[i], label=f'{i}')
                    ax2.plot(profile.rad, profile.mu - magdiff, color=colors[i])

        # Update reference profile as median of normalized profiles for next iteration
        ref_profile.mu = np.nanmedian(np.array(profiles), axis=0)          # Update reference profile as median of normalized profiles
        ref_profile.mu -= ref_profile.mu[0]  # Normalize updated reference profile to its peak
        std = np.nanmean(np.nanstd(np.array(profiles), axis=0))
        
        print(f"{k+1}: Updated reference profile. std of the stacked profile = {std:.3f} mag/arcsec^2")
        methods[methods==0] = 1  # Ensure unsaturated stars now is normalized with method 1 in next iterations to avoid biasing the reference profile

    if plot:
        ax1.set_title('Star Profiles')
        ax1.set_xlabel('Radius (pixels)')
        ax1.set_ylabel(r'$\mu$ (mag/arcsec$^2$)')
        ax1.legend(ncols=4, fontsize=10)
        ax2.set_title('Profiles Normalized')
        ax2.set_xlabel('Radius (pixels)')
        ax2.set_ylabel(r'$\mu$ (mag/arcsec$^2$)')
        ax1.invert_yaxis()
        ax2.invert_yaxis()
        fig.tight_layout()
    return normalized


