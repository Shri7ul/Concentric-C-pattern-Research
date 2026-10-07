"""Thermal physics model (temperature, emissivity, radiance)."""
import numpy as np
import matplotlib.pyplot as plt

SIZE = 500

# Background temperature
BACKGROUND_TEMP = 25.0

# C-shaped target temperature
TARGET_TEMP = 40.0

# Create temperature map
temperature = np.full(
    (SIZE, SIZE),
    BACKGROUND_TEMP,
    dtype=np.float32
)

cx = SIZE // 2
cy = SIZE // 2


def draw_c_temperature(
    temperature,
    cx,
    cy,
    radius,
    width,
    target_temp,
    gap_angle=45
):
    height, width_image = temperature.shape

    y, x = np.ogrid[:height, :width_image]

    # Distance of every pixel from center
    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    # Ring
    ring = (
        (distance >= radius - width / 2)
        &
        (distance <= radius + width / 2)
    )

    # Angle
    angle = np.degrees(
        np.arctan2(y - cy, x - cx)
    )

    angle = (angle + 360) % 360

    # Gap
    gap = (
        (angle >= 0)
        &
        (angle <= gap_angle)
    )

    # C shape
    c_shape = ring & ~gap

    # Assign temperature
    temperature[c_shape] = target_temp

    return temperature


# Outer C
temperature = draw_c_temperature(
    temperature,
    cx,
    cy,
    radius=150,
    width=30,
    target_temp=TARGET_TEMP
)

# Inner C
temperature = draw_c_temperature(
    temperature,
    cx,
    cy,
    radius=75,
    width=20,
    target_temp=TARGET_TEMP
)


# Display temperature map
plt.imshow(
    temperature,
    cmap="inferno",
    vmin=BACKGROUND_TEMP,
    vmax=TARGET_TEMP
)

plt.colorbar(label="Temperature (°C)")
plt.title("Simulated Thermal Image")

plt.show()

temperature = np.full(
    (SIZE, SIZE),
    BACKGROUND_TEMP,
    dtype=np.float32
)

