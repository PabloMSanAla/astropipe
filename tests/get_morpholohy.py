import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from astropipe.utils import morphology


if __name__ == '__main__':

    # Create a blank 100x100 image
    image = np.zeros((100, 100))

    # Define ellipse parameters
    center = (50, 50)  # Center of the ellipse (y, x)
    a_real = 15  # Semi-major axis
    b_real = 6  # Semi-minor axis
    theta_real = 32*np.pi/180  # Rotation angle in radians (0 means no rotation)
    eps_real = 1 -b_real/a_real

    # Create an elliptical aperture
    aperture = EllipticalAperture(center, a_real, b_real, theta_real)
    # Create 
    mask = aperture.to_mask(method='center')
    image = mask.to_image((100, 100))

    angle, a, eps = morphology(image)
    # Print results
    print(f"Semi-major axis (a): {a_real}-->{a:.2f}")
    print(f"Semi-minor axis (b): {b_real}-->{a*(1-eps):.2f}")
    print(f'Angle:               {theta_real*180/np.pi}-->{angle:.2f}')

    # Display the image
    # plt.figure()
    # plt.imshow(image)








