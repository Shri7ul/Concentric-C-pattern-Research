"""Generate thermal test targets (e.g. 4-bar / MRTD style)."""
import numpy as np
import matplotlib.pyplot as plt


# Image size
SIZE = 500

# Create an empty image
image = np.zeros((SIZE, SIZE), dtype=np.uint8)

# Center of the image
cx = SIZE // 2
cy = SIZE // 2

# Outer C parameters
outer_radius = 150
outer_width = 30

# Inner C parameters
inner_radius = 75
inner_width = 20

def draw_c(image, cx, cy, radius, width, gap_angle=45):
    """
    Draw a C-shaped ring into the image.
    """

    height, width_image = image.shape

    y, x = np.ogrid[:height, :width_image]

    # Distance from the center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Ring region
    ring = (
        (distance >= radius - width / 2)
        &
        (distance <= radius + width / 2)
    )

    # Angle of every pixel
    angle = np.degrees(np.arctan2(y - cy, x - cx))

    # Normalize angle to 0-360
    angle = (angle + 360) % 360

    # Remove a section to create the C gap
    gap_start = 0
    gap_end = gap_angle

    gap = (
        (angle >= gap_start)
        &
        (angle <= gap_end)
    )

    c_shape = ring & ~gap

    image[c_shape] = 255

    return image


image = draw_c(
    image,
    cx,
    cy,
    outer_radius,
    outer_width
)

image = draw_c(
    image,
    cx,
    cy,
    inner_radius,
    inner_width
)

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()