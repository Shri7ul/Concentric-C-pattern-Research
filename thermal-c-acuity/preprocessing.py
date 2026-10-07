# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def temperature_to_uint8(
#     temperature,
#     min_temp,
#     max_temp
# ):
#     """
#     Convert temperature values to 8-bit intensity.
#     """

#     normalized = (
#         (temperature - min_temp)
#         / (max_temp - min_temp)
#     )

#     normalized = np.clip(
#         normalized,
#         0,
#         1
#     )

#     image = (
#         normalized * 255
#     ).astype(np.uint8)

#     return image


# def threshold_thermal_image(
#     image,
#     threshold
# ):
#     """
#     Convert thermal image into binary image.
#     """

#     _, binary = cv2.threshold(
#         image,
#         threshold,
#         255,
#         cv2.THRESH_BINARY
#     )

#     return binary


# def clean_binary_image(
#     binary_image,
#     kernel_size=3
# ):
#     """
#     Remove small noise using morphological opening.
#     """

#     kernel = np.ones(
#         (kernel_size, kernel_size),
#         np.uint8
#     )

#     cleaned = cv2.morphologyEx(
#         binary_image,
#         cv2.MORPH_OPEN,
#         kernel
#     )

#     return cleaned


# # ==================================================
# # Create actual concentric C temperature map
# # ==================================================

# WIDTH = 320
# HEIGHT = 256

# BACKGROUND_TEMP = 25.0
# TARGET_TEMP = 40.0

# temperature = np.full(
#     (HEIGHT, WIDTH),
#     BACKGROUND_TEMP,
#     dtype=np.float32
# )

# cx = WIDTH // 2
# cy = HEIGHT // 2


# def draw_c(
#     temperature,
#     cx,
#     cy,
#     radius,
#     width,
#     target_temp,
#     gap_angle=45
# ):

#     height, width_image = temperature.shape

#     y, x = np.ogrid[
#         :height,
#         :width_image
#     ]

#     distance = np.sqrt(
#         (x - cx) ** 2 +
#         (y - cy) ** 2
#     )

#     ring = (
#         (distance >= radius - width / 2)
#         &
#         (distance <= radius + width / 2)
#     )

#     angle = np.degrees(
#         np.arctan2(
#             y - cy,
#             x - cx
#         )
#     )

#     angle = (angle + 360) % 360

#     gap = (
#         (angle >= 0)
#         &
#         (angle <= gap_angle)
#     )

#     c_shape = ring & ~gap

#     temperature[c_shape] = target_temp

#     return temperature


# # Outer C
# temperature = draw_c(
#     temperature,
#     cx,
#     cy,
#     radius=30,
#     width=6,
#     target_temp=TARGET_TEMP
# )

# # Inner C
# temperature = draw_c(
#     temperature,
#     cx,
#     cy,
#     radius=15,
#     width=4,
#     target_temp=TARGET_TEMP
# )


# # ==================================================
# # Convert temperature → image
# # ==================================================

# thermal_image = temperature_to_uint8(
#     temperature,
#     min_temp=25,
#     max_temp=40
# )


# # ==================================================
# # Threshold
# # ==================================================

# binary_image = threshold_thermal_image(
#     thermal_image,
#     threshold=80
# )


# # ==================================================
# # Morphological cleanup
# # ==================================================

# cleaned_image = clean_binary_image(
#     binary_image,
#     kernel_size=3
# )


# # ==================================================
# # Display
# # ==================================================

# plt.figure(figsize=(12, 4))


# plt.subplot(1, 3, 1)

# plt.imshow(
#     thermal_image,
#     cmap="inferno"
# )

# plt.title("Thermal Image")
# plt.axis("off")


# plt.subplot(1, 3, 2)

# plt.imshow(
#     binary_image,
#     cmap="gray"
# )

# plt.title("Binary Image")
# plt.axis("off")


# plt.subplot(1, 3, 3)

# plt.imshow(
#     cleaned_image,
#     cmap="gray"
# )

# plt.title("Cleaned Image")
# plt.axis("off")


# plt.tight_layout()

# plt.show()




import cv2
import numpy as np


# ==================================================
# Temperature → 8-bit image
# ==================================================

def temperature_to_uint8(
    temperature,
    min_temp,
    max_temp
):
    """
    Convert temperature values to 8-bit intensity.

    min_temp -> 0
    max_temp -> 255
    """

    normalized = (
        (temperature - min_temp)
        / (max_temp - min_temp)
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    image = (
        normalized * 255
    ).astype(np.uint8)

    return image


# ==================================================
# Threshold
# ==================================================

def threshold_thermal_image(
    image,
    threshold
):
    """
    Convert thermal image into binary image.
    """

    _, binary = cv2.threshold(
        image,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    return binary


# ==================================================
# Morphological cleanup
# ==================================================

def clean_binary_image(
    binary_image,
    kernel_size=3
):
    """
    Remove small noise using morphological opening.
    """

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned


# ==================================================
# Complete preprocessing pipeline
# ==================================================

def preprocess(
    temperature,
    min_temp=25.0,
    max_temp=40.0,
    threshold=80,
    kernel_size=3
):
    """
    Complete thermal image preprocessing pipeline.

    Input:
        temperature -> 2D temperature map

    Output:
        cleaned binary image
    """

    # Step 1:
    # Temperature → 8-bit image
    thermal_image = temperature_to_uint8(
        temperature,
        min_temp,
        max_temp
    )

    # Step 2:
    # Thermal image → binary image
    binary_image = threshold_thermal_image(
        thermal_image,
        threshold
    )

    # Step 3:
    # Remove small noise
    cleaned_image = clean_binary_image(
        binary_image,
        kernel_size
    )

    return cleaned_image