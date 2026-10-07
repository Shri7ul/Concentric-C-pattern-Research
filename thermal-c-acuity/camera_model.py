import numpy as np
import matplotlib.pyplot as plt
import cv2

from noise_model import add_temperature_noise

# ==================================================
# Camera parameters
# ==================================================

CAMERA_WIDTH = 320
CAMERA_HEIGHT = 256

PIXEL_PITCH = 12e-6       # 12 micrometers
FOCAL_LENGTH = 6e-3       # 6 millimeters


# ==================================================
# Target physical dimensions
# ==================================================

OUTER_RADIUS_MM = 60
OUTER_WIDTH_MM = 12

INNER_RADIUS_MM = 30
INNER_WIDTH_MM = 8


BACKGROUND_TEMP = 25.0
TARGET_TEMP = 40.0


# ==================================================
# Calculate IFOV
# ==================================================

IFOV = PIXEL_PITCH / FOCAL_LENGTH

print("IFOV:", IFOV, "radians")
print("IFOV:", IFOV * 1000, "mrad")


# ==================================================
# Convert physical size to pixels
# ==================================================

def mm_per_pixel(distance_m):

    return distance_m * 1000 * IFOV


def mm_to_pixels(size_mm, distance_m):

    resolution = mm_per_pixel(distance_m)

    return size_mm / resolution


# ==================================================
# Draw C
# ==================================================

def draw_c(
    image,
    cx,
    cy,
    radius,
    width,
    target_temp,
    gap_angle=45
):

    height, width_image = image.shape

    y, x = np.ogrid[:height, :width_image]

    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    ring = (
        (distance >= radius - width / 2)
        &
        (distance <= radius + width / 2)
    )

    angle = np.degrees(
        np.arctan2(y - cy, x - cx)
    )

    angle = (angle + 360) % 360

    gap = (
        (angle >= 0)
        &
        (angle <= gap_angle)
    )

    c_shape = ring & ~gap

    image[c_shape] = target_temp

    return image


# ==================================================
# Simulate target at a distance
# ==================================================

def simulate_target(distance_m):

    temperature = np.full(
        (CAMERA_HEIGHT, CAMERA_WIDTH),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    cx = CAMERA_WIDTH // 2
    cy = CAMERA_HEIGHT // 2

    # Convert physical dimensions to pixels
    outer_radius = mm_to_pixels(
        OUTER_RADIUS_MM,
        distance_m
    )

    outer_width = mm_to_pixels(
        OUTER_WIDTH_MM,
        distance_m
    )

    inner_radius = mm_to_pixels(
        INNER_RADIUS_MM,
        distance_m
    )

    inner_width = mm_to_pixels(
        INNER_WIDTH_MM,
        distance_m
    )

    # Draw outer C
    temperature = draw_c(
        temperature,
        cx,
        cy,
        outer_radius,
        outer_width,
        TARGET_TEMP
    )

    # Draw inner C
    temperature = draw_c(
        temperature,
        cx,
        cy,
        inner_radius,
        inner_width,
        TARGET_TEMP
    )

    return temperature


# ==================================================
# Test different distances
# ==================================================

distance = 1.0

camera_image = simulate_target(distance) #without blur

# Apply optical blur
camera_image = cv2.GaussianBlur(
    camera_image,
    (5, 5),
    sigmaX=0.0
)

camera_image = add_temperature_noise(
    camera_image,
    noise_std=2.0
)

# ==================================================
# Display
# ==================================================

plt.figure(figsize=(8, 5))

plt.imshow(
    camera_image,
    cmap="inferno",
    vmin=BACKGROUND_TEMP,
    vmax=TARGET_TEMP
)

plt.colorbar(label="Temperature (°C)")

plt.title(
    f"Thermal Camera — Distance = {distance} m"
)

plt.show()

