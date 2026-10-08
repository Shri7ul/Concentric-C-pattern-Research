"""
Experiment 2:
Blur-only thermal C acuity test.

Purpose:
Measure the effect of image blur alone.

Distance : FIXED
Noise    : OFF
Blur     : VARIABLE
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from c_detector import (
    create_two_level_target,
    detect_concentric_pair,
    BACKGROUND_TEMP,
    TARGET_TEMP,
    THRESHOLD,
    KERNEL_SIZE
)

from preprocessing import preprocess


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

WIDTH = 320
HEIGHT = 256

DISTANCE = 0.5

# Gaussian blur kernel sizes.
#
# IMPORTANT:
# Gaussian kernel must be odd.
#
# 1  = no blur
# 3  = very small blur
# 5  = small blur
# 7  = moderate blur
# 9  = stronger blur
# 11 = strong blur
# 15 = very strong blur

BLUR_KERNELS = [
    1,
    3,
    5,
    7,
    9,
    11,
    15,
    21
]


# ============================================================
# 2. APPLY GAUSSIAN BLUR
# ============================================================

def apply_blur(
    temperature,
    kernel_size
):
    """
    Apply Gaussian blur to the temperature image.

    kernel_size = 1 means NO blur.
    """

    if kernel_size == 1:

        return temperature.copy()

    blurred = cv2.GaussianBlur(
        temperature,
        (
            kernel_size,
            kernel_size
        ),
        sigmaX=0
    )

    return blurred


# ============================================================
# 3. RUN C DETECTOR
# ============================================================

def detect_target(
    temperature
):
    """
    Preprocess the thermal image and
    detect a two-level concentric C.
    """

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    cleaned_image = preprocess(
        temperature,
        min_temp=BACKGROUND_TEMP,
        max_temp=TARGET_TEMP,
        threshold=THRESHOLD,
        kernel_size=KERNEL_SIZE
    )

    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        cleaned_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Detect concentric pair
    # --------------------------------------------------------

    result = detect_concentric_pair(
        cleaned_image,
        contours
    )

    detected = (
        result is not None
    )

    return (
        detected,
        cleaned_image,
        contours,
        result
    )


# ============================================================
# 4. RUN EXPERIMENT
# ============================================================

def run_experiment():

    print()
    print("=" * 65)
    print("EXPERIMENT 2 — BLUR ONLY")
    print("=" * 65)

    print()
    print(f"Distance : {DISTANCE} m")
    print("Noise    : OFF")
    print("Blur     : VARIABLE")
    print()

    # --------------------------------------------------------
    # Create original 2-level C
    # --------------------------------------------------------

    original_temperature = (
        create_two_level_target()
    )

    results = []

    # --------------------------------------------------------
    # Test every blur level
    # --------------------------------------------------------

    for kernel_size in BLUR_KERNELS:

        # ----------------------------------------------------
        # Apply blur
        # ----------------------------------------------------

        blurred_temperature = apply_blur(
            original_temperature,
            kernel_size
        )

        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        detected, cleaned, contours, result = (
            detect_target(
                blurred_temperature
            )
        )

        # ----------------------------------------------------
        # Detection rate
        # ----------------------------------------------------

        if detected:

            detection_rate = 100

            status = "DETECTED"

        else:

            detection_rate = 0

            status = "NOT DETECTED"

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "kernel": kernel_size,
            "detected": detected,
            "detection_rate": detection_rate
        })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if kernel_size == 1:

            blur_name = "NO BLUR"

        else:

            blur_name = (
                f"{kernel_size}x{kernel_size}"
            )

        print(
            f"Blur = {blur_name:<12}"
            f" →  {status}"
        )

    return results


# ============================================================
# 5. PLOT RESULT
# ============================================================

def plot_results(
    results
):

    kernels = [
        item["kernel"]
        for item in results
    ]

    detection_rates = [
        item["detection_rate"]
        for item in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        kernels,
        detection_rates,
        marker="o"
    )

    plt.xlabel(
        "Gaussian Blur Kernel Size"
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.title(
        "Thermal C Detection vs Blur"
    )

    plt.ylim(
        -5,
        105
    )

    plt.xticks(
        kernels
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 6. SHOW VISUAL EFFECT OF BLUR
# ============================================================

def show_examples():

    original_temperature = (
        create_two_level_target()
    )

    example_kernels = [
        1,
        3,
        5,
        7,
        11,
        21
    ]

    plt.figure(
        figsize=(12, 8)
    )

    for i, kernel_size in enumerate(
        example_kernels
    ):

        blurred = apply_blur(
            original_temperature,
            kernel_size
        )

        plt.subplot(
            2,
            3,
            i + 1
        )

        plt.imshow(
            blurred,
            cmap="inferno"
        )

        if kernel_size == 1:

            title = "No Blur"

        else:

            title = (
                f"{kernel_size} × {kernel_size}"
            )

        plt.title(
            title
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Effect of Gaussian Blur on Thermal C"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. PRINT SUMMARY
# ============================================================

def print_summary(
    results
):

    print()
    print("=" * 65)
    print("BLUR EXPERIMENT SUMMARY")
    print("=" * 65)

    print()
    print(
        f"{'Blur Kernel':<15}"
        f"{'Detection':<15}"
    )

    print(
        "-" * 30
    )

    for item in results:

        kernel = item["kernel"]

        if kernel == 1:

            name = "No blur"

        else:

            name = f"{kernel}x{kernel}"

        if item["detected"]:

            result = "DETECTED"

        else:

            result = "NOT DETECTED"

        print(
            f"{name:<15}"
            f"{result:<15}"
        )


# ============================================================
# 8. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    results = run_experiment()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Visual examples
    # --------------------------------------------------------

    show_examples()


# ============================================================
# 9. RUN
# ============================================================

if __name__ == "__main__":

    main()