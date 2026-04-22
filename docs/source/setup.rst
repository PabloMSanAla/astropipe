Installation
============

Requirements
------------

AstroPipe requires Python 3.8+ and the following packages (installed
automatically via pip):

- ``astropy``
- ``numpy``
- ``scipy``
- ``matplotlib``
- ``photutils``
- ``astroquery``
- ``fabada``

Optional but recommended for the full masking pipeline:

- `SExtractor <https://www.astromatic.net/software/sextractor/>`_ (system
  package, e.g. ``sudo apt install source-extractor``)

Installing AstroPipe
--------------------

Clone the repository and install in editable mode::

   git clone https://github.com/PabloMSanAla/astropipe.git
   cd astropipe
   pip install -e .

Or, using the provided conda environment file::

   conda env create -f environment.yml
   conda activate astropipe
   pip install -e .

Quick check
-----------

.. code-block:: python

   import astropipe
   print(astropipe.__version__)
