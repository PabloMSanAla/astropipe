Tutorial: Surface Brightness Profiles with AstroPipe
=====================================================

This tutorial walks through a complete surface brightness profile analysis using
AstroPipe on a CAVITY galaxy. It mirrors the workflow in the
``demos/cavity.ipynb`` notebook.

.. note::
   The full interactive notebook is available at ``demos/cavity.ipynb``.

Prerequisites
-------------

Install AstroPipe and its dependencies (see :doc:`Installation <setup>`).
You also need SExtractor installed on your system for the masking step. A CSV
catalogue with galaxy coordinates is supplied in ``demos/cavity_demo.csv``.

1. Downloading the Image
------------------------

Use :func:`astropipe.query.legacy_mosaic` to download a cutout from the
`Legacy Survey DR10 <https://www.legacysurvey.org/>`_ for the galaxy of
interest. The function skips the download when the file already exists.

.. code-block:: python

   import os
   import numpy as np
   from os.path import join
   from astropy.table import Table
   from astropipe.query import legacy_mosaic

   index = 0                               # change to select a different galaxy
   tbl   = Table.read('cavity_demo.csv')
   name  = tbl[index]['galaxy']
   ra, dec = tbl[index][['RA', 'DEC']]
   size  = 0.15                            # cutout size in degrees
   outdir = 'temp/'

   filename = join(outdir, f'{name}_ls.fits')
   if not os.path.isdir(outdir):
       os.mkdir(outdir)

   if not os.path.isfile(filename):
       legacy_mosaic(ra, dec, outdir=outdir, name=filename,
                     width=size, bands='r', verbose=True)

2. Creating the Image Object
-----------------------------

Wrap the downloaded FITS file in an :class:`astropipe.classes.Image` object.
Pass the zero-point (``zp``) needed to convert counts to magnitudes, then set
the object coordinates so that morphology and profile routines know which source
to analyse.

.. code-block:: python

   from astropipe.classes import Image

   image = Image(filename, hdu=0, zp=22.5)
   image.obj(ra, dec)
   image.name = name

   print(f'pixel scale: {image.pixel_scale:.4f}  '
         f'object position [x,y]: {image.pix}')
   ax = image.show(width=500, vmin=21, vmax=26, cmap='nipy_spectral')

3. Masking Sources
------------------

:func:`astropipe.masking.sexmask` produces an accurate mask by running
SExtractor three times with different configurations and applying a FABADA
denoised residual step to catch point sources embedded in the galaxy disc.
If SExtractor is unavailable, :func:`astropipe.masking.fastmask` provides a
pure-Python fallback.

.. code-block:: python

   from astropy.io import fits
   from astropipe.classes import Directories
   from astropipe.masking import sexmask, fastmask

   folders = Directories(name, path=os.path.dirname(filename))

   if not os.path.isfile(folders.mask):
       sexmask(image, folders)
       # Pure-Python fallback (no SExtractor required):
       # mask = fastmask(image.data, (image.x, image.y), nsigma=1.0, fwhm=3)
       # fits.PrimaryHDU(mask, image.header).writeto(folders.mask, overwrite=True)
       # image.set_mask(mask)
   else:
       image.set_mask(fits.getdata(folders.mask))

   image.show(width=500, vmin=26, vmax=18, cmap='nipy_spectral')

4. Morphology and Sky Background
---------------------------------

:meth:`~astropipe.classes.Image.get_morphology` fits elliptical isophotes to
non-masked pixels to derive the positional angle (``pa``), ellipticity
(``eps``), and an indicative galaxy radius (``reff``).

:meth:`~astropipe.classes.Image.get_background` then constructs a growth curve
to locate the sky plateau and estimates the background via both an elliptical
aperture and random rectangular apertures.

.. code-block:: python

   image.get_morphology()
   image.get_background(growth_rate=1.05,
                        out=join(folders.out, f'{name}_bkg.jpg'))

   print(f'PA = {image.pa:.2f} deg  |  eps = {image.eps:.2f}')
   print(f'Background = {image.bkg:.2e}  |  '
         f'Std = {image.bkgstd:.2e}  |  Radius = {image.bkgrad:.2e}')

5. Surface Brightness Profiles
--------------------------------

**Isophotal photometry** — fits ellipse parameters at each radial step
(Jedrzejewski 1987):

.. code-block:: python

   from astropipe.profile import Profile

   image.reff = 10   # starting semi-major axis in pixels
   if not os.path.isfile(folders.profile):
       profile = image.isophotal_photometry(max_r=1.1 * image.bkgrad)
       profile.write(folders.profile)
   else:
       profile = Profile(filename=folders.profile)
       profile.brightness()

   fig = profile.plot()

**Fixed-ellipse photometry** — forces constant morphological parameters
across all radii:

.. code-block:: python

   fix_profile = image.radial_photometry(max_r=1.2 * image.bkgrad,
                                         growth_rate=1.1)
   fix_profile.plot()

6. Publication-Ready Figures
------------------------------

:func:`astropipe.plotting.surface_figure` combines the image, mask, and profile
into a single multi-panel figure suitable for publications:

.. code-block:: python

   from astropipe.plotting import surface_figure

   fig = surface_figure(image, profile, vmin=20, vmax=26,
                        out=join(folders.out, f'{name}_photometry.jpg'))

7. Photometric Parameters
--------------------------

The :class:`~astropipe.profile.Profile` class provides methods for common
photometric measurements:

.. code-block:: python

   from matplotlib import pyplot as plt

   sma, cog = profile.curveOfGrowth()
   mag_total = profile.totalMagnitude(sma, cog)

   results = {
       'mag':     mag_total,
       'reff':    profile.fractionalRadius(mag_total, fluxFrac=0.5),
       'sb_eff':  profile.surfaceBrightness(
                      profile.fractionalRadius(mag_total, fluxFrac=0.5)),
       'c82':     profile.concentration(mag_total),
       'c31':     (profile.fractionalRadius(mag_total, fluxFrac=0.75) /
                   profile.fractionalRadius(mag_total, fluxFrac=0.25)),
       'rpetro':  profile.petrosianRadius(),
       'r25p5':   profile.isophotalRadius(25.5),
       'r26p5':   profile.isophotalRadius(26.5),
       'r28':     profile.isophotalRadius(28),
   }

   for k, v in results.items():
       print(f'{k:>8s} = {v:.3f}')

Troubleshooting
---------------

Editing the mask interactively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`astropipe.plotting.MaskEditor` provides an interactive Qt5 widget to
add or remove masked regions::

   import matplotlib
   matplotlib.use('Qt5Agg')
   %matplotlib qt5

   from astropipe.plotting import MaskEditor
   from PyQt5.QtWidgets import QApplication
   import sys

   app = QApplication([])
   editor = MaskEditor(image, vmin=20, vmax=26)
   sys.exit(app.exec_())

Isophote fitting fails (NO MEANINGFUL FIT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduce ``image.reff`` to start the fit in a high-SNR inner region:

.. code-block:: python

   image.reff = 10   # smaller starting radius
   profile = image.isophotal_photometry(max_r=1.1 * image.bkgrad)

Adjusting the sky background manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   profile.bkg    = -1.6e-4   # manual sky value
   profile.bkgstd =  5.0e-4   # manual sky standard deviation
   profile.brightness()
   profile.plot()

Extending the profile to a larger radius
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   profile.extend(image.data, 1.8 * image.bkgrad)
   profile.plot()
