#!/usr/bin/env python3
"""
Interactive Surface Brightness Profile Creator

This script provides a GUI-based workflow for creating surface brightness
profiles from astronomical images with different masking procedures.

Workflow:
1. Select FITS image file
2. Input object coordinates (RA, Dec)
3. Choose masking method (SExtractor, MTObjects, or Fast)
4. Edit mask interactively
5. Extract surface brightness profile
6. Display results

Author: Based on astropipe framework
"""

import os
import sys
from pathlib import Path

import numpy as np

# GUI imports
try:
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("Error: PyQt5 is required. Install with: pip install PyQt5")
    sys.exit(1)

# Astropipe imports
try:
    import matplotlib
    from astropipe.classes import AstroGNU, Directories, Image
    from astropipe.masking import fastmask, mtomask, sexmask
    from astropipe.plotting import MaskEditor, show, surface_figure
    from astropy.io import fits
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
    from matplotlib.backends.backend_qt5agg import (
        NavigationToolbar2QT as NavigationToolbar,
    )
    from matplotlib.figure import Figure
    from matplotlib.patches import Ellipse

    matplotlib.use("Qt5Agg")
except ImportError as e:
    print(f"Error importing astropipe modules: {e}")
    print("Make sure astropipe is installed and in your Python path")
    sys.exit(1)


class MaskingThread(QThread):
    """Thread for running masking operations without blocking GUI"""

    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, img, folders, method, fwhm=None):
        super().__init__()
        self.img = img
        self.folders = folders
        self.method = method
        self.fwhm = fwhm

    def run(self):
        try:
            self.progress.emit(f"Running {self.method} masking...")

            if self.method == "SExtractor":
                success = sexmask(
                    self.img,
                    self.folders,
                    fwhm=self.fwhm,
                    plot=True,
                    temp=False,
                )
            elif self.method == "MTObjects":
                success = mtomask(self.img, self.folders, plot=True)
            elif self.method == "Fast":
                mask = fastmask(self.img.data, self.img.pix, nsigma=1, fwhm=5)
                self.img.set_mask(mask)
                # Save mask
                mask_array = np.array(mask, dtype=np.uint8)
                fits.PrimaryHDU(mask_array, header=self.img.header).writeto(
                    self.folders.mask, overwrite=True
                )
                success = True
            elif self.method == "NoiseChisel":
                # Create AstroGNU instance
                astrognu = AstroGNU(self.img.data, self.folders)
                # Run noisechisel and segment methods
                astrognu.noisechisel(config="--numthreads=8", keep=True)
                astrognu.segment(
                    config="--numthreads=8", keep=True, clumps=True
                )
                mask_array = astrognu.objects
                mask_array[self.img.y, self.img.x] = 0  # Unmask object
                fits.PrimaryHDU(mask_array, header=self.img.header).writeto(
                    self.folders.mask, overwrite=True
                )
                success = True
            else:
                success = False

            if success:
                self.progress.emit("Masking completed successfully!")
                self.finished.emit(True, "")
            else:
                self.finished.emit(False, "Masking failed")

        except Exception as e:
            self.finished.emit(False, str(e))


class FitProfileThread(QThread):
    """Thread for running isophote fitting without blocking GUI"""

    finished = pyqtSignal(bool, str, object)
    progress = pyqtSignal(str)

    def __init__(self, img, growth_rate, max_r):
        super().__init__()
        self.img = img
        self.growth_rate = growth_rate
        self.max_r = max_r

    def run(self):
        try:
            import traceback

            self.progress.emit("Fitting isophotes...")
            failed = False
            try:
                # Run isophote fitting using img.isophotal_photometry
                profile = self.img.isophotal_photometry(
                    growth_rate=self.growth_rate,
                    max_r=self.max_r,
                )
            except:
                failed = True
                self.progress.emit(
                    "Isophote fitting failed, iterating until convergence"
                )

            if failed:
                original_pa = self.img.pa
                original_eps = self.img.eps
                original_reff = self.img.reff
                random_tries = np.column_stack(
                    [
                        np.abs(np.random.normal(0.8, 0.1, (100, 2))),
                        np.abs(np.random.normal(0, 0.2, 100)),
                    ]
                )
                for i, k, j in random_tries:
                    try:
                        self.img.pa = i * original_pa
                        self.img.eps = k * original_eps
                        self.img.reff = j * original_reff

                        profile = self.img.isophotal_photometry(
                            growth_rate=self.growth_rate,
                            max_r=self.max_r,
                        )
                        # If successful, break out of loop
                        break
                    except:
                        # Continue to next iteration if it fails
                        continue

            self.progress.emit("Isophote fitting completed!")
            self.finished.emit(True, "", profile)

        except Exception as e:
            error_details = f"{str(e)}\n{traceback.format_exc()}"
            self.progress.emit(f"ERROR: {str(e)}")
            self.finished.emit(False, error_details, None)


class SurfaceBrightnessProfileGUI(QMainWindow):
    """Main GUI for interactive surface brightness profile creation"""

    def __init__(self):
        super().__init__()
        self.img = None
        self.folders = None
        self.profile = None
        self.mask_editor = None
        self.isophote_table = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Interactive Surface Brightness Profile Creator")
        self.setGeometry(100, 100, 1400, 800)

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel for plotting
        right_panel = self.create_plot_panel()
        main_layout.addWidget(right_panel, stretch=3)

        # Title
        title = QLabel("Surface Brightness Profile Creator")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        # Step 1: Image Selection & Configuration
        step1_group = self.create_image_selection_group()
        left_layout.addWidget(step1_group)

        # Step 2: Masking Options
        step2_group = self.create_masking_group()
        left_layout.addWidget(step2_group)

        # Step 3: Morphology & Background
        step3_group = self.create_morphology_background_group()
        left_layout.addWidget(step3_group)

        # Step 4: Profile Extraction
        step4_group = self.create_profile_group()
        left_layout.addWidget(step4_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Log/Status area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        left_layout.addWidget(QLabel("Status Log:"))
        left_layout.addWidget(self.log_text)

        # Initialize state
        self.update_button_states()

    def create_plot_panel(self):
        """Create right panel for plotting with tabs"""
        from PyQt5.QtWidgets import QTabWidget

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)

        # Create tab widget
        self.plot_tabs = QTabWidget()
        plot_layout.addWidget(self.plot_tabs)

        # Tab 1: Image/Mask
        self.image_tab = QWidget()
        image_layout = QVBoxLayout(self.image_tab)

        self.image_figure = Figure(figsize=(10, 8))
        self.image_canvas = FigureCanvas(self.image_figure)
        self.image_toolbar = NavigationToolbar(
            self.image_canvas, self.image_tab
        )

        image_layout.addWidget(self.image_toolbar)
        image_layout.addWidget(self.image_canvas)

        # Image plot controls
        self.image_controls_widget = QWidget()
        image_controls_layout = QHBoxLayout(self.image_controls_widget)

        image_controls_layout.addWidget(QLabel("vmin:"))
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setRange(0, 35)
        self.vmin_spin.setDecimals(1)
        self.vmin_spin.setValue(21.0)
        image_controls_layout.addWidget(self.vmin_spin)

        image_controls_layout.addWidget(QLabel("vmax:"))
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setRange(0, 35)
        self.vmax_spin.setDecimals(1)
        self.vmax_spin.setValue(28.5)
        image_controls_layout.addWidget(self.vmax_spin)

        image_controls_layout.addWidget(QLabel("Width (pix):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(50, 2000)
        self.width_spin.setDecimals(0)
        self.width_spin.setValue(400)
        image_controls_layout.addWidget(self.width_spin)

        update_image_btn = QPushButton("Update Plot")
        update_image_btn.clicked.connect(self.refresh_image_plot)
        image_controls_layout.addWidget(update_image_btn)

        image_controls_layout.addStretch()
        self.image_controls_widget.setVisible(False)
        image_layout.addWidget(self.image_controls_widget)

        # Initialize image plot
        ax = self.image_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Load an image to begin",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.image_canvas.draw()

        # Tab 2: Background
        self.background_tab = QWidget()
        background_layout = QVBoxLayout(self.background_tab)

        self.background_figure = Figure(figsize=(10, 8))
        self.background_canvas = FigureCanvas(self.background_figure)
        self.background_toolbar = NavigationToolbar(
            self.background_canvas, self.background_tab
        )

        background_layout.addWidget(self.background_toolbar)
        background_layout.addWidget(self.background_canvas)

        # Initialize background plot
        ax = self.background_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Compute background to see plot",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.background_canvas.draw()

        # Tab 3: Profile
        self.profile_tab = QWidget()
        profile_layout = QVBoxLayout(self.profile_tab)

        self.profile_figure = Figure(figsize=(10, 8))
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self.profile_toolbar = NavigationToolbar(
            self.profile_canvas, self.profile_tab
        )

        profile_layout.addWidget(self.profile_toolbar)
        profile_layout.addWidget(self.profile_canvas)

        # Initialize profile plot
        ax = self.profile_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Extract profile to see plot",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.profile_canvas.draw()

        # Tab 4: Surface Profile Results
        self.surface_tab = QWidget()
        surface_layout = QVBoxLayout(self.surface_tab)

        self.surface_figure = Figure(figsize=(10, 8))
        self.surface_canvas = FigureCanvas(self.surface_figure)
        self.surface_toolbar = NavigationToolbar(
            self.surface_canvas, self.surface_tab
        )

        surface_layout.addWidget(self.surface_toolbar)
        surface_layout.addWidget(self.surface_canvas)

        # Initialize surface plot
        ax = self.surface_figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Show results to see surface profile",
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        self.surface_canvas.draw()

        # Add tabs
        self.plot_tabs.addTab(self.image_tab, "Image/Mask")
        self.plot_tabs.addTab(self.background_tab, "Background")
        self.plot_tabs.addTab(self.profile_tab, "Profile")
        self.plot_tabs.addTab(self.surface_tab, "Surface Profile")

        # Disable background, profile, and surface tabs initially
        self.plot_tabs.setTabEnabled(1, False)
        self.plot_tabs.setTabEnabled(2, False)
        self.plot_tabs.setTabEnabled(3, False)

        # Track current plot state
        self.current_plot_type = None

        return plot_widget

    def update_plot_image(self):
        """Update plot to show image after loading"""
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)

        # Show image using astropipe's show method with the axis
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        width = int(self.width_spin.value())

        self.img.show(ax=ax, vmin=vmin, vmax=vmax, width=width)
        self.image_canvas.draw()

        # Show plot controls and track current plot type
        self.image_controls_widget.setVisible(True)
        self.current_plot_type = "image"

    def update_plot_morphology(self):
        """Update plot to show image with morphology overlay"""
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)

        # Show image with adjustable parameters
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        width = int(self.width_spin.value())

        self.img.show(ax=ax, vmin=vmin, vmax=vmax, width=width, plotmask=False)

        # Overlay morphology ellipse
        if (
            hasattr(self.img, "reff")
            and hasattr(self.img, "pa")
            and hasattr(self.img, "eps")
        ):
            # Calculate ellipse dimensions
            width = 2 * self.img.reff * self.img.pixel_scale
            height = (
                2 * self.img.reff * (1 - self.img.eps) * self.img.pixel_scale
            )

            # Create ellipse centered at object
            ellipse = Ellipse(
                xy=(0, 0),  # Centered in show() coordinate system
                width=width,
                height=height,
                angle=self.img.pa,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label=f"PA={self.img.pa:.1f}°, ε={self.img.eps:.2f}",
            )
            ax.add_patch(ellipse)
            ax.legend(loc="upper right", fontsize=10)

        ax.set_title("Image with Morphology", fontsize=14, fontweight="bold")
        self.image_canvas.draw()

    def update_plot_mask(self):
        """Update plot to show image with mask overlay"""
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)

        # Show image with mask
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        width = int(self.width_spin.value())

        self.img.show(ax=ax, vmin=vmin, vmax=vmax, width=width, plotmask=True)

        self.image_canvas.draw()

        # Show plot controls and track current plot type
        self.image_controls_widget.setVisible(True)
        self.current_plot_type = "mask"

    def update_plot_mask_with_morphology(self):
        """Update plot to show image with mask and morphology overlay"""
        self.image_figure.clear()
        ax = self.image_figure.add_subplot(111)

        # Show image with mask
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        width = int(self.width_spin.value())

        self.img.show(ax=ax, vmin=vmin, vmax=vmax, width=width, plotmask=True)

        # Overlay morphology ellipse
        if (
            hasattr(self.img, "reff")
            and hasattr(self.img, "pa")
            and hasattr(self.img, "eps")
        ):
            # Calculate ellipse dimensions
            width = self.img.reff * self.img.pixel_scale
            height = self.img.reff * (1 - self.img.eps) * self.img.pixel_scale

            # Create ellipse centered at object
            ellipse = Ellipse(
                xy=(
                    self.img.x,
                    self.img.y,
                ),  # Centered in show() coordinate system
                width=width,
                height=height,
                angle=self.img.pa,
                edgecolor="red",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label=f"PA={self.img.pa:.1f}°, ε={self.img.eps:.2f}",
            )
            ax.add_patch(ellipse)
            ax.legend(loc="upper right", fontsize=10)

        self.image_canvas.draw()

        # Show plot controls and track current plot type
        self.image_controls_widget.setVisible(True)
        self.current_plot_type = "mask_morphology"

    def update_plot_profile(self):
        """Update plot to show the extracted profile"""
        self.profile_figure.clear()

        # Create 3 subplots for profile: mu, PA, eps
        ax_mu = self.profile_figure.add_subplot(3, 1, 1)
        ax_pa = self.profile_figure.add_subplot(3, 1, 2, sharex=ax_mu)
        ax_eps = self.profile_figure.add_subplot(3, 1, 3, sharex=ax_mu)

        # Plot using the profile's plot method
        self.profile.plot(axes=(ax_mu, ax_pa, ax_eps))

        self.profile_figure.tight_layout()
        self.profile_canvas.draw()

        # Enable profile tab and switch to it
        self.plot_tabs.setTabEnabled(2, True)
        self.plot_tabs.setCurrentIndex(2)
        self.current_plot_type = "profile"

    def refresh_image_plot(self):
        """Refresh the image/mask plot with updated parameters"""
        if self.current_plot_type == "image":
            self.update_plot_image()
        elif self.current_plot_type == "mask":
            self.update_plot_mask()
        elif self.current_plot_type == "mask_morphology":
            self.update_plot_mask_with_morphology()

    def update_plot_background(self):
        """Update plot to show background profile"""
        self.background_figure.clear()

        if os.path.exists(self.background_out):
            # Load the saved background image
            import matplotlib.image as mpimg

            img_bg = mpimg.imread(self.background_out)

            ax = self.background_figure.add_subplot(111)
            ax.imshow(img_bg)
            ax.axis("off")  # Hide axes for cleaner display

            self.background_canvas.draw()

            # Enable background tab and switch to it
            self.plot_tabs.setTabEnabled(1, True)
            self.plot_tabs.setCurrentIndex(1)

    def create_image_selection_group(self):
        """Create image selection group"""
        group = QGroupBox("Step 1: Load Image & Set Coordinates")
        layout = QGridLayout()

        # File selection
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("No file selected...")
        self.file_path_edit.setReadOnly(True)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)

        # Image parameters
        self.zp_spin = QDoubleSpinBox()
        self.zp_spin.setRange(0, 50)
        self.zp_spin.setDecimals(2)
        self.zp_spin.setValue(22.5)

        self.hdu_spin = QDoubleSpinBox()
        self.hdu_spin.setRange(0, 10)
        self.hdu_spin.setDecimals(0)
        self.hdu_spin.setValue(1)

        # Object coordinates
        self.ra_spin = QDoubleSpinBox()
        self.ra_spin.setRange(0, 360)
        self.ra_spin.setDecimals(7)
        self.ra_spin.setValue(195.2415670)

        self.dec_spin = QDoubleSpinBox()
        self.dec_spin.setRange(-90, 90)
        self.dec_spin.setDecimals(7)
        self.dec_spin.setValue(26.9763683)

        self.load_btn = QPushButton("Load Image & Set Object")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setEnabled(False)

        # Layout
        layout.addWidget(QLabel("File:"), 0, 0)
        layout.addWidget(self.file_path_edit, 0, 1, 1, 3)
        layout.addWidget(browse_btn, 0, 4)

        layout.addWidget(QLabel("Zero Point:"), 1, 0)
        layout.addWidget(self.zp_spin, 1, 1)
        layout.addWidget(QLabel("HDU:"), 1, 2)
        layout.addWidget(self.hdu_spin, 1, 3)

        layout.addWidget(QLabel("RA (deg):"), 2, 0)
        layout.addWidget(self.ra_spin, 2, 1)
        layout.addWidget(QLabel("Dec (deg):"), 2, 2)
        layout.addWidget(self.dec_spin, 2, 3)

        layout.addWidget(self.load_btn, 3, 0, 1, 5)

        group.setLayout(layout)
        return group

    def create_masking_group(self):
        """Create masking options group"""
        group = QGroupBox("Step 2: Create & Edit Mask")
        layout = QGridLayout()

        self.mask_method_combo = QComboBox()
        self.mask_method_combo.addItems(
            ["Fast", "SExtractor", "MTObjects", "NoiseChisel"]
        )

        self.fwhm_spin = QDoubleSpinBox()
        self.fwhm_spin.setRange(1, 50)
        self.fwhm_spin.setDecimals(1)
        self.fwhm_spin.setValue(5.0)

        self.create_mask_btn = QPushButton("Create Mask")
        self.create_mask_btn.clicked.connect(self.create_mask)
        self.create_mask_btn.setEnabled(False)

        self.edit_mask_btn = QPushButton("Edit Mask")
        self.edit_mask_btn.clicked.connect(self.edit_mask)
        self.edit_mask_btn.setEnabled(False)

        self.load_mask_btn = QPushButton("Load Existing Mask")
        self.load_mask_btn.clicked.connect(self.load_existing_mask)
        self.load_mask_btn.setEnabled(False)

        # Mask path display
        self.mask_path_label = QLabel("Mask: Not loaded")
        self.mask_path_label.setWordWrap(True)
        self.mask_path_label.setStyleSheet(
            "QLabel { color: gray; font-size: 10px; }"
        )

        layout.addWidget(QLabel("Method:"), 0, 0)
        layout.addWidget(self.mask_method_combo, 0, 1)
        layout.addWidget(QLabel("FWHM:"), 0, 2)
        layout.addWidget(self.fwhm_spin, 0, 3)
        layout.addWidget(self.create_mask_btn, 1, 0, 1, 2)
        layout.addWidget(self.edit_mask_btn, 1, 2)
        layout.addWidget(self.load_mask_btn, 1, 3)
        layout.addWidget(self.mask_path_label, 2, 0, 1, 4)

        group.setLayout(layout)
        return group

    def create_morphology_background_group(self):
        """Create morphology and background group"""
        group = QGroupBox("Step 3: Morphology & Background")
        layout = QGridLayout()

        # Morphology button and nsigma parameter
        self.get_morphology_btn = QPushButton("Get Morphology")
        self.get_morphology_btn.clicked.connect(self.get_morphology)
        self.get_morphology_btn.setEnabled(False)

        self.nsigma_spin = QDoubleSpinBox()
        self.nsigma_spin.setRange(0.1, 10.0)
        self.nsigma_spin.setDecimals(1)
        self.nsigma_spin.setValue(1.0)
        self.nsigma_spin.setPrefix("nsigma: ")
        self.nsigma_spin.setEnabled(False)

        # Morphology parameters
        self.pa_spin = QDoubleSpinBox()
        self.pa_spin.setRange(-180, 180)
        self.pa_spin.setDecimals(2)
        self.pa_spin.setValue(0.0)
        self.pa_spin.setEnabled(False)
        self.pa_spin.editingFinished.connect(self.update_morphology_params)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0, 0.99)
        self.eps_spin.setDecimals(3)
        self.eps_spin.setValue(0.0)
        self.eps_spin.setEnabled(False)
        self.eps_spin.editingFinished.connect(self.update_morphology_params)

        self.reff_spin = QDoubleSpinBox()
        self.reff_spin.setRange(1, 1000)
        self.reff_spin.setDecimals(2)
        self.reff_spin.setValue(10.0)
        self.reff_spin.setEnabled(False)
        self.reff_spin.editingFinished.connect(self.update_morphology_params)

        # Background button and init parameter
        self.get_background_btn = QPushButton("Get Background")
        self.get_background_btn.clicked.connect(self.get_background)
        self.get_background_btn.setEnabled(False)

        self.bkg_init_spin = QDoubleSpinBox()
        self.bkg_init_spin.setRange(1, 500)
        self.bkg_init_spin.setDecimals(0)
        self.bkg_init_spin.setValue(30.0)
        self.bkg_init_spin.setPrefix("init: ")
        self.bkg_init_spin.setEnabled(False)

        # Background parameters
        self.bkg_spin = QDoubleSpinBox()
        self.bkg_spin.setRange(-1e6, 1e6)
        self.bkg_spin.setDecimals(6)
        self.bkg_spin.setValue(0.0)
        self.bkg_spin.setEnabled(False)
        self.bkg_spin.editingFinished.connect(self.update_background_params)

        self.bkgstd_spin = QDoubleSpinBox()
        self.bkgstd_spin.setRange(0, 1e6)
        self.bkgstd_spin.setDecimals(6)
        self.bkgstd_spin.setValue(0.0)
        self.bkgstd_spin.setEnabled(False)
        self.bkgstd_spin.editingFinished.connect(self.update_background_params)

        self.bkgrad_spin = QDoubleSpinBox()
        self.bkgrad_spin.setRange(1, 5000)
        self.bkgrad_spin.setDecimals(2)
        self.bkgrad_spin.setValue(100.0)
        self.bkgrad_spin.setEnabled(False)
        self.bkgrad_spin.editingFinished.connect(self.update_background_params)

        # Layout
        layout.addWidget(self.get_morphology_btn, 0, 0)
        layout.addWidget(self.nsigma_spin, 0, 1)
        layout.addWidget(QLabel("PA (°):"), 1, 0)
        layout.addWidget(self.pa_spin, 1, 1)
        layout.addWidget(QLabel("ε:"), 2, 0)
        layout.addWidget(self.eps_spin, 2, 1)
        layout.addWidget(QLabel("R_eff (pix):"), 3, 0)
        layout.addWidget(self.reff_spin, 3, 1)

        layout.addWidget(self.get_background_btn, 0, 2)
        layout.addWidget(self.bkg_init_spin, 0, 3)
        layout.addWidget(QLabel("Bkg:"), 1, 2)
        layout.addWidget(self.bkg_spin, 1, 3)
        layout.addWidget(QLabel("Bkg STD:"), 2, 2)
        layout.addWidget(self.bkgstd_spin, 2, 3)
        layout.addWidget(QLabel("Bkg Rad (pix):"), 3, 2)
        layout.addWidget(self.bkgrad_spin, 3, 3)

        group.setLayout(layout)
        return group

    def create_profile_group(self):
        """Create profile extraction group"""
        group = QGroupBox("Step 4: Fit & Extract Profile")
        layout = QGridLayout()

        # Parameters
        self.growth_rate_spin = QDoubleSpinBox()
        self.growth_rate_spin.setRange(1.01, 2.0)
        self.growth_rate_spin.setDecimals(2)
        self.growth_rate_spin.setValue(1.08)

        self.max_r_factor_spin = QDoubleSpinBox()
        self.max_r_factor_spin.setRange(0.5, 5.0)
        self.max_r_factor_spin.setDecimals(1)
        self.max_r_factor_spin.setValue(1.5)

        self.fit_limit_spin = QDoubleSpinBox()
        self.fit_limit_spin.setRange(0.01, 5.0)
        self.fit_limit_spin.setDecimals(2)
        self.fit_limit_spin.setValue(0.1)

        self.smooth_spin = QDoubleSpinBox()
        self.smooth_spin.setRange(0, 10)
        self.smooth_spin.setDecimals(1)
        self.smooth_spin.setValue(3)

        # Buttons
        self.fit_profile_btn = QPushButton("Fit Profile")
        self.fit_profile_btn.clicked.connect(self.fit_profile)
        self.fit_profile_btn.setEnabled(False)

        self.extract_profile_btn = QPushButton("Measure Profile")
        self.extract_profile_btn.clicked.connect(self.extract_profile)
        self.extract_profile_btn.setEnabled(False)

        self.load_profile_btn = QPushButton("Load Profile")
        self.load_profile_btn.clicked.connect(self.load_existing_profile)
        self.load_profile_btn.setEnabled(False)

        self.show_results_btn = QPushButton("Show Results")
        self.show_results_btn.clicked.connect(self.show_results)
        self.show_results_btn.setEnabled(False)

        self.save_results_btn = QPushButton("Save Figure")
        self.save_results_btn.clicked.connect(self.save_results)
        self.save_results_btn.setEnabled(False)

        # Profile path display
        self.profile_path_label = QLabel("Profile: Not extracted")
        self.profile_path_label.setWordWrap(True)
        self.profile_path_label.setStyleSheet(
            "QLabel { color: gray; font-size: 10px; }"
        )

        # Layout - Parameters
        layout.addWidget(QLabel("Growth Rate:"), 0, 0)
        layout.addWidget(self.growth_rate_spin, 0, 1)
        layout.addWidget(QLabel("Max R (×bkgrad):"), 0, 2)
        layout.addWidget(self.max_r_factor_spin, 0, 3)
        layout.addWidget(QLabel("Fit Limit:"), 1, 0)
        layout.addWidget(self.fit_limit_spin, 1, 1)
        layout.addWidget(QLabel("Smooth:"), 1, 2)
        layout.addWidget(self.smooth_spin, 1, 3)

        # Layout - Buttons
        layout.addWidget(self.fit_profile_btn, 2, 0, 1, 2)
        layout.addWidget(self.extract_profile_btn, 2, 2, 1, 2)
        layout.addWidget(self.load_profile_btn, 3, 0, 1, 2)
        layout.addWidget(self.show_results_btn, 3, 2)
        layout.addWidget(self.save_results_btn, 3, 3)
        layout.addWidget(self.profile_path_label, 4, 0, 1, 4)

        group.setLayout(layout)
        return group

    def log(self, message):
        """Add message to log"""
        self.log_text.append(message)
        QApplication.processEvents()

    def browse_file(self):
        """Browse for FITS file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FITS Image",
            "",
            "FITS Files (*.fits *.fit);;All Files (*)",
        )
        if file_path:
            self.file_path_edit.setText(file_path)
            self.load_btn.setEnabled(True)
            self.log(f"Selected file: {os.path.basename(file_path)}")

    def load_image(self):
        """Load FITS image and set object coordinates"""
        try:
            file_path = self.file_path_edit.text()
            zp = self.zp_spin.value()
            hdu = int(self.hdu_spin.value())
            ra = self.ra_spin.value()
            dec = self.dec_spin.value()

            self.log(f"Loading image from HDU {hdu}...")
            self.img = Image(file_path, zp=zp, hdu=hdu)
            self.img.data[np.isnan(self.img.data)] = 0

            # Create directories
            base_name = Path(file_path).stem
            self.folders = Directories(base_name)

            self.log(f"Image loaded: {self.img.data.shape}")
            self.log(f"Pixel scale: {self.img.pixel_scale:.3f} arcsec/pixel")

            # Set object coordinates
            self.log(f"Setting object at RA={ra:.6f}, Dec={dec:.6f}")
            self.img.obj(ra, dec)
            self.log(f"Object pixel position: ({self.img.x}, {self.img.y})")

            self.create_mask_btn.setEnabled(True)
            self.load_mask_btn.setEnabled(True)
            self.load_profile_btn.setEnabled(True)
            self.update_button_states()

            # Update plot to show image
            self.update_plot_image()

            QMessageBox.information(
                self, "Success", "Image loaded and object set successfully!"
            )

        except Exception as e:
            self.log(f"ERROR loading image: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load image:\n{str(e)}"
            )

    def get_morphology(self):
        """Get morphology parameters"""
        try:
            nsigma = self.nsigma_spin.value()
            self.log(f"Calculating morphology with nsigma={nsigma}...")
            self.img.get_morphology(nsigma=nsigma)

            self.log(f"PA: {self.img.pa:.2f}°")
            self.log(f"Ellipticity: {self.img.eps:.3f}")
            self.log(f"Effective radius: {self.img.reff:.2f} pixels")

            # Populate morphology spinboxes
            self.pa_spin.blockSignals(True)
            self.eps_spin.blockSignals(True)
            self.reff_spin.blockSignals(True)

            self.pa_spin.setValue(self.img.pa)
            self.eps_spin.setValue(self.img.eps)
            self.reff_spin.setValue(self.img.reff)

            self.pa_spin.blockSignals(False)
            self.eps_spin.blockSignals(False)
            self.reff_spin.blockSignals(False)

            # Enable morphology parameter editing
            self.pa_spin.setEnabled(True)
            self.eps_spin.setEnabled(True)
            self.reff_spin.setEnabled(True)

            # Enable background button and init parameter after morphology is done
            self.get_background_btn.setEnabled(True)
            self.bkg_init_spin.setEnabled(True)

            # Check if both morphology and background are done
            if hasattr(self.img, "bkg") and hasattr(self.img, "bkgrad"):
                self.fit_profile_btn.setEnabled(True)
                self.extract_profile_btn.setEnabled(True)
            self.update_button_states()

            # Update plot to show mask with morphology overlay
            self.update_plot_mask_with_morphology()

            QMessageBox.information(
                self,
                "Morphology",
                f"PA: {self.img.pa:.2f}°\n"
                f"Ellipticity: {self.img.eps:.3f}\n"
                f"R_eff: {self.img.reff:.2f} pix",
            )

        except Exception as e:
            self.log(f"ERROR calculating morphology: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to get morphology:\n{str(e)}"
            )

    def get_background(self):
        """Get background estimation"""
        try:
            init_value = int(self.bkg_init_spin.value())
            self.log(f"Estimating background with init={init_value}...")

            self.background_out = os.path.join(
                self.folders.temp, f"{self.img.name}_background.png"
            )
            self.img.get_background(init=init_value, out=self.background_out)

            self.log(f"Background: {self.img.bkg:.6e}")
            self.log(f"Background STD: {self.img.bkgstd:.6e}")
            self.log(f"Background radius: {self.img.bkgrad:.2f} pixels")

            # Populate background spinboxes
            self.bkg_spin.blockSignals(True)
            self.bkgstd_spin.blockSignals(True)
            self.bkgrad_spin.blockSignals(True)

            self.bkg_spin.setValue(self.img.bkg)
            self.bkgstd_spin.setValue(self.img.bkgstd)
            self.bkgrad_spin.setValue(self.img.bkgrad)

            self.bkg_spin.blockSignals(False)
            self.bkgstd_spin.blockSignals(False)
            self.bkgrad_spin.blockSignals(False)

            # Enable background parameter editing
            self.bkg_spin.setEnabled(True)
            self.bkgstd_spin.setEnabled(True)
            self.bkgrad_spin.setEnabled(True)

            # Check if both morphology and background are done
            if (
                hasattr(self.img, "pa")
                and hasattr(self.img, "eps")
                and hasattr(self.img, "reff")
            ):
                self.fit_profile_btn.setEnabled(True)
                self.extract_profile_btn.setEnabled(True)
            self.update_button_states()

            # Update background plot and enable tab
            self.update_plot_background()

            QMessageBox.information(
                self,
                "Background",
                f"Background: {self.img.bkg:.6e}\n"
                f"STD: {self.img.bkgstd:.6e}\n"
                f"Radius: {self.img.bkgrad:.2f} pix",
            )

        except Exception as e:
            self.log(f"ERROR estimating background: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to estimate background:\n{str(e)}"
            )

    def update_morphology_params(self):
        """Update image morphology parameters from spinboxes"""
        if hasattr(self.img, "pa"):
            self.img.pa = self.pa_spin.value()
            self.img.eps = self.eps_spin.value()
            self.img.reff = self.reff_spin.value()
            self.log(
                f"Morphology updated: PA={self.img.pa:.2f}°, ε={self.img.eps:.3f}, R_eff={self.img.reff:.2f} pix"
            )

            # Update plot if showing morphology
            if self.current_plot_type == "mask_morphology":
                self.update_plot_mask_with_morphology()

    def update_background_params(self):
        """Update image background parameters from spinboxes"""
        if hasattr(self.img, "bkg"):
            self.img.bkg = self.bkg_spin.value()
            self.img.bkgstd = self.bkgstd_spin.value()
            self.img.bkgrad = self.bkgrad_spin.value()
            self.log(
                f"Background updated: bkg={self.img.bkg:.6e}, std={self.img.bkgstd:.6e}, rad={self.img.bkgrad:.2f} pix"
            )

            # Update background plot if tab is enabled
            if self.plot_tabs.isTabEnabled(1):
                try:
                    self.background_figure.clear()
                    from astropipe.plotting import surface_figure

                    surface_figure(self.img, self.background_figure)
                    self.background_canvas.draw()
                except Exception as e:
                    self.log(
                        f"WARNING: Could not update background plot: {str(e)}"
                    )

            # If profile exists, recompute brightness and update plot
            if self.profile is not None and hasattr(self.profile, "rad"):
                try:
                    self.log(
                        "Recomputing surface brightness with new background..."
                    )

                    self.profile.bkg = self.img.bkg
                    self.profile.bkgstd = self.img.bkgstd
                    self.profile.brightness()

                    # Update profile plot
                    self.update_plot_profile()

                    self.log(f"Profile updated with new background parameters")

                except Exception as e:
                    self.log(f"WARNING: Could not update profile: {str(e)}")

    def create_mask(self):
        """Create mask using selected method"""
        try:
            method = self.mask_method_combo.currentText()
            fwhm = self.fwhm_spin.value() if method == "SExtractor" else None

            self.log(f"Creating mask using {method} method...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate

            # Disable buttons during processing
            self.create_mask_btn.setEnabled(False)

            # Create and start thread
            self.masking_thread = MaskingThread(
                self.img, self.folders, method, fwhm
            )
            self.masking_thread.progress.connect(self.log)
            self.masking_thread.finished.connect(self.on_masking_finished)
            self.masking_thread.start()

        except Exception as e:
            self.log(f"ERROR creating mask: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to create mask:\n{str(e)}"
            )
            self.progress_bar.setVisible(False)
            self.create_mask_btn.setEnabled(True)

    def on_masking_finished(self, success, error_msg):
        """Handle masking completion"""
        self.progress_bar.setVisible(False)
        self.create_mask_btn.setEnabled(True)

        if success:
            # Load the mask
            if os.path.exists(self.folders.mask):
                mask_data = fits.getdata(self.folders.mask)
                self.img.set_mask(mask_data)
                self.log("Mask loaded successfully!")

                # Update mask path label
                self.mask_path_label.setText(f"Mask: {self.folders.mask}")
                self.mask_path_label.setToolTip(self.folders.mask)

                self.edit_mask_btn.setEnabled(True)
                self.get_morphology_btn.setEnabled(True)
                self.nsigma_spin.setEnabled(True)
                self.update_button_states()

                # Update plot to show mask
                self.update_plot_mask()

                QMessageBox.information(
                    self, "Success", "Mask created successfully!"
                )
            else:
                self.log("WARNING: Mask file not found")
        else:
            self.log(f"ERROR: {error_msg}")
            QMessageBox.critical(
                self, "Error", f"Masking failed:\n{error_msg}"
            )

    def load_existing_mask(self):
        """Load an existing mask file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Mask File",
                "",
                "FITS Files (*.fits *.fit);;All Files (*)",
            )
            if file_path:
                self.log(f"Loading mask from {os.path.basename(file_path)}...")
                mask_data = fits.getdata(file_path)
                self.img.set_mask(mask_data)

                # Update mask path label
                self.mask_path_label.setText(f"Mask: {file_path}")
                self.mask_path_label.setToolTip(file_path)

                self.edit_mask_btn.setEnabled(True)
                self.get_morphology_btn.setEnabled(True)
                self.nsigma_spin.setEnabled(True)
                self.update_button_states()

                # Update plot to show mask
                self.update_plot_mask()

                self.log("Mask loaded successfully!")
                QMessageBox.information(
                    self, "Success", "Mask loaded successfully!"
                )

        except Exception as e:
            self.log(f"ERROR loading mask: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load mask:\n{str(e)}"
            )

    def edit_mask(self):
        """Open mask editor"""
        try:
            self.log("Opening mask editor...")

            # Calculate vmin and vmax for display
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()

            # Create mask editor
            self.mask_editor = MaskEditor(self.img, vmin=vmin, vmax=vmax)
            self.mask_editor.show()

            self.log("Mask editor opened. Close it when done editing.")

        except Exception as e:
            self.log(f"ERROR opening mask editor: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to open mask editor:\n{str(e)}"
            )

    def fit_profile(self):
        """Fit isophotes to image"""
        try:
            growth_rate = self.growth_rate_spin.value()
            max_r_factor = self.max_r_factor_spin.value()
            max_r = max_r_factor * self.img.bkgrad

            self.log(f"Fitting isophotes (growth={growth_rate})...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate

            # Disable buttons during processing
            self.fit_profile_btn.setEnabled(False)
            self.extract_profile_btn.setEnabled(False)

            # Create and start thread
            self.fit_thread = FitProfileThread(self.img, growth_rate, max_r)
            self.fit_thread.progress.connect(self.log)
            self.fit_thread.finished.connect(self.on_fit_finished)
            self.fit_thread.start()

        except Exception as e:
            import traceback

            error_details = f"{str(e)}\n{traceback.format_exc()}"
            self.log(f"ERROR fitting profile: {error_details}")
            QMessageBox.critical(
                self, "Error", f"Failed to fit profile:\n{error_details}"
            )
            self.progress_bar.setVisible(False)
            self.fit_profile_btn.setEnabled(True)

    def on_fit_finished(self, success, error_msg, profile):
        """Handle fit completion"""
        self.progress_bar.setVisible(False)
        self.fit_profile_btn.setEnabled(True)

        if success:
            self.profile = profile
            self.log(
                f"Isophote fitting completed with {len(profile.rad)} points"
            )

            # Save profile automatically
            profile_path = (
                self.folders.profile
                if hasattr(self, "folders")
                else f"{self.img.name}_profile.ecsv"
            )
            self.profile.write(profile_path)
            self.log(f"Profile saved to: {profile_path}")

            # Update profile path label
            self.profile_path_label.setText(f"Profile: {profile_path}")
            self.profile_path_label.setToolTip(profile_path)

            # Enable result buttons and measure profile button
            self.extract_profile_btn.setEnabled(True)
            self.show_results_btn.setEnabled(True)
            self.save_results_btn.setEnabled(True)
            self.update_button_states()

            # Update plot to show profile
            self.update_plot_profile()

            QMessageBox.information(
                self,
                "Success",
                f"Isophote fitting completed!\n{len(profile.rad)} radial bins\nSaved to: {os.path.basename(profile_path)}",
            )
        else:
            self.log(f"ERROR: {error_msg}")
            QMessageBox.critical(
                self, "Error", f"Isophote fitting failed:\n{error_msg}"
            )

    def extract_profile(self):
        """Extract radial profile from fitted isophotes or directly"""
        try:
            import traceback

            growth_rate = self.growth_rate_spin.value()
            max_r_factor = self.max_r_factor_spin.value()
            max_r = max_r_factor * self.img.bkgrad

            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate
            self.extract_profile_btn.setEnabled(False)

            QApplication.processEvents()  # Update UI

            if self.profile is not None and hasattr(self.profile, "rad"):
                # Use fitted isophotes
                try:
                    from astropipe.profile import elliptical_radial_profile

                    self.log(
                        f"Measuring profile from fitted isophotes (max_r={max_r:.1f} pix)..."
                    )

                    self.profile = elliptical_radial_profile(
                        self.img.data,
                        self.profile.rad,
                        (
                            self.profile.x.value,
                            self.profile.y.value,
                        ),
                        self.profile.pa.value,
                        self.profile.eps.value,
                    )
                except Exception as e:
                    self.log(
                        f"WARNING: elliptical_radial_profile failed: {str(e)}"
                    )
                    self.log("Falling back to direct radial photometry...")
                    # Fall back to direct method
                    self.profile = None

            if self.profile is None or not hasattr(self.profile, "rad"):
                # Use direct radial photometry
                self.log(
                    f"Extracting profile (growth={growth_rate}, max_r={max_r:.1f} pix)..."
                )

                self.profile = self.img.radial_photometry(
                    max_r=max_r,
                    growth_rate=growth_rate,
                )

            self.log(f"Profile extracted with {len(self.profile.rad)} points")

            # Save profile automatically
            profile_path = (
                self.folders.profile
                if hasattr(self, "folders")
                and hasattr(self.folders, "profile")
                else f"{self.img.name}_profile.ecsv"
            )
            self.profile.write(profile_path)
            self.log(f"Profile saved to: {profile_path}")

            # Update profile path label
            self.profile_path_label.setText(f"Profile: {profile_path}")
            self.profile_path_label.setToolTip(profile_path)

            self.show_results_btn.setEnabled(True)
            self.save_results_btn.setEnabled(True)
            self.update_button_states()

            # Update plot to show profile
            self.update_plot_profile()

            # Hide progress bar
            self.progress_bar.setVisible(False)
            self.extract_profile_btn.setEnabled(True)

            QMessageBox.information(
                self,
                "Success",
                f"Profile extracted successfully!\n{len(self.profile.rad)} radial bins\nSaved to: {os.path.basename(profile_path)}",
            )

        except Exception as e:
            import traceback

            error_details = f"{str(e)}\n{traceback.format_exc()}"
            self.log(f"ERROR extracting profile: {error_details}")

            # Make sure to hide progress bar and re-enable button
            self.progress_bar.setVisible(False)
            self.extract_profile_btn.setEnabled(True)

            QMessageBox.critical(
                self, "Error", f"Failed to extract profile:\n{str(e)}"
            )

    def load_existing_profile(self):
        """Load an existing profile file"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Profile File",
                "",
                "FITS Files (*.fits *.fit);;All Files (*)",
            )
            if file_path:
                from astropipe.profile import Profile

                self.log(
                    f"Loading profile from {os.path.basename(file_path)}..."
                )
                self.profile = Profile(filename=file_path)

                # Update profile path label
                self.profile_path_label.setText(f"Profile: {file_path}")
                self.profile_path_label.setToolTip(file_path)

                self.show_results_btn.setEnabled(True)
                self.save_results_btn.setEnabled(True)
                self.update_button_states()

                # Update plot to show profile
                self.update_plot_profile()

                self.log("Profile loaded successfully!")
                QMessageBox.information(
                    self,
                    "Success",
                    f"Profile loaded successfully!\n{len(self.profile.rad)} radial bins",
                )

        except Exception as e:
            self.log(f"ERROR loading profile: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to load profile:\n{str(e)}"
            )

    def show_results(self):
        """Display surface brightness figure"""
        try:
            self.log("Generating surface brightness figure...")

            # Generate surface_figure (returns a matplotlib figure)
            temp_fig = surface_figure(
                self.img,
                self.profile,
                vmin=self.vmin_spin.value(),
                vmax=self.vmax_spin.value(),
                radmax=self.max_r_factor_spin.value() * self.img.bkgrad,
            )

            # Save to PDF
            self.result_out_pdf = os.path.join(
                self.folders.out, f"{self.img.name}_surface_profile.pdf"
            )
            temp_fig.savefig(
                self.result_out_pdf,
                format="pdf",
                bbox_inches="tight",
                pad_inches=0.1,
            )

            # Save to PNG for display in GUI
            self.result_out_png = os.path.join(
                self.folders.out, f"{self.img.name}_surface_profile.png"
            )
            temp_fig.savefig(
                self.result_out_png,
                format="png",
                dpi=150,
                bbox_inches="tight",
                pad_inches=0.1,
            )

            # Close temporary figure
            plt.close(temp_fig)

            # Clear the surface figure and display the PNG image
            self.surface_figure.clear()

            # Load and display the saved PNG image
            import matplotlib.image as mpimg

            img_result = mpimg.imread(self.result_out_png)

            ax = self.surface_figure.add_subplot(111)
            ax.imshow(img_result)
            ax.axis("off")  # Hide axes for cleaner display

            # Draw the canvas
            self.surface_canvas.draw()

            # Enable surface tab and switch to it
            self.plot_tabs.setTabEnabled(3, True)
            self.plot_tabs.setCurrentIndex(3)

            os.remove(self.result_out_png)  # Clean up temporary PNG file

            self.log("Figure displayed and saved!")

        except Exception as e:
            import traceback

            error_details = f"{str(e)}\n{traceback.format_exc()}"
            self.log(f"ERROR displaying results: {error_details}")
            QMessageBox.critical(
                self, "Error", f"Failed to display results:\n{str(e)}"
            )

    def save_results(self):
        """Save surface brightness figure"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Figure",
                f"{self.img.name}_surface_profile.pdf",
                "PDF Files (*.pdf);;PNG Files (*.png);;All Files (*)",
            )

            if file_path:
                self.log(f"Saving figure to {os.path.basename(file_path)}...")

                fig = surface_figure(
                    self.img,
                    self.profile,
                    vmin=23.5,
                    vmax=29,
                    radmax=1.2 * self.img.bkgrad,
                )

                fig.savefig(
                    file_path,
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.1,
                )
                plt.close(fig)

                self.log(f"Figure saved successfully!")
                QMessageBox.information(
                    self, "Success", f"Figure saved to:\n{file_path}"
                )

        except Exception as e:
            self.log(f"ERROR saving figure: {str(e)}")
            QMessageBox.critical(
                self, "Error", f"Failed to save figure:\n{str(e)}"
            )

    def update_button_states(self):
        """Update button states based on current workflow state"""
        # This helps maintain a logical workflow
        pass


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window
    window = SurfaceBrightnessProfileGUI()
    window.show()

    # # Auto-load configuration for testing/demo
    # # Set the file path
    # auto_file = "/Users/pablom.sanchezalarcon/Downloads/GTC_enanas_matias/DF44/DF44_coadd_g_decCal.fits"
    # auto_mask = "/Users/pablom.sanchezalarcon/Downloads/GTC_enanas_matias/Code/astropipe_DF44_coadd_g_decCal/DF44_coadd_g_decCal_mask.fits"

    # if os.path.exists(auto_file):
    #     # Set file path
    #     window.file_path_edit.setText(auto_file)
    #     window.load_btn.setEnabled(True)

    #     # Load image automatically
    #     QApplication.processEvents()  # Update UI
    #     window.load_image()

    #     # Load mask if it exists
    #     if os.path.exists(auto_mask):
    #         QApplication.processEvents()
    #         window.log("Auto-loading mask...")
    #         try:
    #             mask_data = fits.getdata(auto_mask)
    #             window.img.set_mask(mask_data)
    #             window.mask_path_label.setText(f"Mask: {auto_mask}")
    #             window.mask_path_label.setToolTip(auto_mask)
    #             window.edit_mask_btn.setEnabled(True)
    #             window.get_morphology_btn.setEnabled(True)
    #             window.nsigma_spin.setEnabled(True)
    #             window.update_plot_mask()
    #             window.log("Mask loaded successfully!")
    #         except Exception as e:
    #             window.log(f"ERROR auto-loading mask: {str(e)}")

    #     # Run morphology
    #     QApplication.processEvents()
    #     window.get_morphology()

    #     # Run background
    #     QApplication.processEvents()
    #     window.get_background()

    #     # Run measure profile
    #     QApplication.processEvents()
    #     window.extract_profile()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
