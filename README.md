<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU License][license-shield]][license-url]

<!-- [![LinkedIn][linkedin-shield]][linkedin-url]
PROJECT LOGO
<br />
<div align="center">
  <a href="https://github.com/PabloMSanAla/fabada">
    <img src="fabada-logo.svg" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">astropipe</h3>
<h3 align="center">Pipeline mainly focus on data analysis of Astronomical images.</h3>

  <p align="center">
    astropipe is a pipeline aim to produce reliable surface brightness profiles of galaxies. It has built-in functions to reduce, analyse and visualize astronomical images in general. It is meant to help me analyse all the data for my PhD and being able to share it with other colleagues. 
    <br />
    This is a work in progress, use at your own risk!
    <br />
    <a href="https://github.com/PabloMSanAla/astropipe#documents"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/PabloMSanAla/astropipe#demos">View Demo</a>
    ·
    <a href="https://github.com/PabloMSanAla/astropipe/issues">Report Bug</a>
    ·
    <a href="https://github.com/PabloMSanAla/astropipe/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Library</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#cite">Cite</a></li>
    <!-- <li><a href="#acknowledgments">Acknowledgments</a></li> -->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Library

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

All this code is the result of my Ph.D. The aim of this "library" is to help analyse astronomical images. More specifically, to obtain reliable surface brightness profiles. It can also offer more features to reduce, smooth, fit, and visualize images. It is written in Python and mostly uses common libraries (see the [prerequisites](https://github.com/PabloMSanAla/astropipe#prerequisites)). 

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

Since this is a multiple-purpose library it depends on different Python libraries and astronomical software. It also depends on the modules you would like to use. Here I explained briefly the biggest dependencies of the different modules. 


### Prerequisites
####  External Python software
To use the masking module you need to have different external software for Astronomical Image Segmentation. Specifically the following software:

- [SExtractor](https://www.astromatic.net/software/sextractor/)
- [Gnuastro](https://www.gnu.org/software/gnuastro/)
- [MTObjects](https://github.com/CarolineHaigh/mtobjects)

and the aliases need to be stored in your environment variable so you can call the different software. In particular, for Gnuastro, it uses NoiseChisel. This needs to be installed if you want to create masks. However there are different methods that use different software, so it's not mandatory to have all installed but at least, one of them. 

#### Python libraries
The library is written in Python and uses different libraries. Most of them can be installed using pip. You can find the main requisites in the [*pyproject.toml*](https://github.com/PabloMSanAla/astropipe/blob/main/pyproject.toml). 


### Installation

You can install the library and all the external dependencies using a [*conda*](https://www.anaconda.com/) environment as follows:

```sh
  git clone https://github.com/PabloMSanAla/astropipe.git
  cd astropipe
  conda env create -f environment.yml
```

This will create an environment named *astropipe*, which you can activate with:

```sh
  conda activate astropipe
```

Once activated, all functionalities of the astropipe library are available.
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

## Documents

Work in progress...


<!-- Results Paper -->

## Demos

I created different Jupyter notebooks to help you use the pipeline and get a sense of the methods built-in.

- [Cavity](https://github.com/PabloMSanAla/astropipe/blob/main/demos/cavity.ipynb): Jupyter Notebook to create masks, profiles and visualize galaxies from a [CAVITY](https://www.ugr.es/~isa/) field.
  

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTRIBUTING -->

## Contributing

Contributions are what makes the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the GNU General Public License. See [`LICENSE.txt`](https://github.com/PabloMSanAla/astropipe/blob/master/LICENSE) for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Pablo M Sánchez-Alarcón - pmsa.astro@gmail.com

Project Link: [https://github.com/PabloMSanAla/astropipe](https://github.com/PabloMSanAla/astropipe)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CITE -->

## Cite ?

Thank you for using astropipe.


<p align="right">(<a href="#top">back to top</a>)</p>

Readme file taken from [Best README Template](https://github.com/othneildrew/Best-README-Template).

<!-- ACKNOWLEDGMENTS
## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#top">back to top</a>)</p> -->

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/PabloMSanAla/fabada.svg?style=plastic&logo=appveyor
[contributors-url]: https://github.com/PabloMSanAla/astropipe/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/PabloMSanAla/astropipe.svg?style=plastic&logo=appveyor
[forks-url]: https://github.com/PabloMSanAla/astropipe/network/members
[stars-shield]: https://img.shields.io/github/stars/PabloMSanAla/astropipe.svg?style=plastic&logo=appveyor
[stars-url]: https://github.com/PabloMSanAla/astropipe/stargazers
[issues-shield]: https://img.shields.io/github/issues/PabloMSanAla/astropipe.svg?style=plastic&logo=appveyor
[issues-url]: https://github.com/PabloMSanAla/astropipe/issues
[license-shield]: https://img.shields.io/github/license/PabloMSanAla/astropipe.svg?style=plastic&logo=appveyor
[license-url]: https://github.com/PabloMSanAla/astropipe/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=plastic&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[image_results]: src/images/bubble_fabada_24.63dB.jpg
[spectra_results]: src/images/arp256_fabada_28.22dB.jpg
[astronomy_results]: src/images/SDSS_example.jpg
