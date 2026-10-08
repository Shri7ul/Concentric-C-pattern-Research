"""
Experiment 4:
Contrast-only thermal C acuity test.

Distance : FIXED
Blur     : OFF
Noise    : OFF
Background temperature : FIXED
Target temperature     : VARIABLE

Purpose:
Measure how thermal contrast affects
two-level concentric C detection.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from c_detector import (
    detect_concentric_pair
)

from preprocessing import preprocess


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

DISTANCE = 0.5

BACKGROUND_TEMP = 25.0

TRIALS_PER_CONTRAST = 20

# Target temperatures
TARGET_TEMPERATURES = [
    40.0,
    38.0,
    36.0,
    34.0,
    32.0,
    30.0,
    29.0,
    28.0,
    27.0,
    26.0
]


# ============================================================
# 2. TARGET GEOMETRY
# ============================================================

WIDTH = 320
HEIGHT = 256

CX = WIDTH // 2
CY = HEIGHT // 2


# ============================================================
# 3. CREATE TEMPERATURE MAP
# ============================================================

def create_temperature_map(
    target_temp
):
    """
    Create a two-level concentric C
    temperature map.

    Background = 25°C
    Target     = target_temp
    """

    temperature = np.full(
        (HEIGHT, WIDTH),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Coordinate grid
    # --------------------------------------------------------

    y, x = np.ogrid[
        :HEIGHT,
        :WIDTH
    ]

    # --------------------------------------------------------
    # Distance from center
    # --------------------------------------------------------

    distance = np.sqrt(
        (x - CX) ** 2 +
        (y - CY) ** 2
    )

    # --------------------------------------------------------
    # Angle
    # --------------------------------------------------------

    angle = np.degrees(
        np.arctan2(
            y - CY,
            x - CX
        )
    )

    angle = (
        angle + 360
    ) % 360

    # --------------------------------------------------------
    # Gap
    # --------------------------------------------------------

    gap_angle = 40

    gap = (
        (angle >= 0) &
        (angle <= gap_angle)
    )

    # ========================================================
    # OUTER C
    # ========================================================

    outer_radius = 30
    outer_width = 6

    outer_ring = (
        (distance >= outer_radius - outer_width / 2)
        &
        (distance <= outer_radius + outer_width / 2)
    )

    outer_c = (
        outer_ring &
        ~gap
    )

    # ========================================================
    # INNER C
    # ========================================================

    inner_radius = 15
    inner_width = 4

    inner_ring = (
        (distance >= inner_radius - inner_width / 2)
        &
        (distance <= inner_radius + inner_width / 2)
    )

    inner_c = (
        inner_ring &
        ~gap
    )

    # --------------------------------------------------------
    # Put target temperature
    # --------------------------------------------------------

    temperature[
        outer_c | inner_c
    ] = target_temp

    return temperature


# ============================================================
# 4. TEMPERATURE → UINT8
# ============================================================

def temperature_to_uint8(
    temperature,
    target_temp
):
    """
    Convert temperature into an 8-bit thermal image.

    The threshold is calculated from the midpoint
    between background and target temperature.

    This keeps the experiment focused on contrast,
    rather than using a fixed threshold that would
    automatically fail at low target temperatures.
    """

    # --------------------------------------------------------
    # Use fixed physical range
    # --------------------------------------------------------

    min_temp = BACKGROUND_TEMP
    max_temp = 40.0

    normalized = (
        temperature - min_temp
    ) / (
        max_temp - min_temp
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    image = (
        normalized * 255
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Calculate midpoint threshold
    # --------------------------------------------------------

    midpoint_temp = (
        BACKGROUND_TEMP +
        target_temp
    ) / 2

    threshold = (
        (midpoint_temp - min_temp)
        /
        (max_temp - min_temp)
    ) * 255

    threshold = int(
        threshold
    )

    return image, threshold


# ============================================================
# 5. RUN DETECTOR
# ============================================================

def detect_target(
    temperature,
    target_temp
):
    """
    Convert temperature map to thermal image,
    preprocess and detect concentric C.
    """

    # --------------------------------------------------------
    # Temperature → image
    # --------------------------------------------------------

    thermal_image, threshold = (
        temperature_to_uint8(
            temperature,
            target_temp
        )
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    cleaned_image = preprocess(
        temperature,
        min_temp=BACKGROUND_TEMP,
        max_temp=40.0,
        threshold=threshold,
        kernel_size=3
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
    # Detect two-level concentric C
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
        thermal_image,
        cleaned_image,
        contours,
        result
    )


# ============================================================
# 6. TEST ONE TARGET TEMPERATURE
# ============================================================

def test_target_temperature(
    target_temp
):
    """
    Run multiple trials at one target temperature.
    """

    detected_count = 0

    example_image = None

    # --------------------------------------------------------
    # Repeat trials
    # --------------------------------------------------------

    for trial in range(
        TRIALS_PER_CONTRAST
    ):

        # ----------------------------------------------------
        # Create target
        # ----------------------------------------------------

        temperature = (
            create_temperature_map(
                target_temp
            )
        )

        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        (
            detected,
            thermal_image,
            cleaned_image,
            contours,
            result
        ) = detect_target(
            temperature,
            target_temp
        )

        # ----------------------------------------------------
        # Count
        # ----------------------------------------------------

        if detected:

            detected_count += 1

        # ----------------------------------------------------
        # Save example
        # ----------------------------------------------------

        if trial == 0:

            example_image = (
                thermal_image.copy()
            )

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    detection_rate = (
        detected_count /
        TRIALS_PER_CONTRAST
    ) * 100

    # --------------------------------------------------------
    # Thermal contrast
    # --------------------------------------------------------

    contrast = (
        target_temp -
        BACKGROUND_TEMP
    )

    return {
        "target_temp": target_temp,
        "contrast": contrast,
        "detected": detected_count,
        "trials": TRIALS_PER_CONTRAST,
        "detection_rate": detection_rate,
        "example_image": example_image
    }


# ============================================================
# 7. RUN COMPLETE EXPERIMENT
# ============================================================

def run_experiment():

    print()
    print("=" * 65)
    print("EXPERIMENT 4 — CONTRAST ONLY")
    print("=" * 65)

    print()
    print(
        f"Distance : {DISTANCE} m"
    )

    print(
        "Blur     : OFF"
    )

    print(
        "Noise    : OFF"
    )

    print(
        f"Background : {BACKGROUND_TEMP:.1f} °C"
    )

    print(
        f"Trials   : {TRIALS_PER_CONTRAST}"
    )

    print()

    results = []

    # --------------------------------------------------------
    # Test every target temperature
    # --------------------------------------------------------

    for target_temp in TARGET_TEMPERATURES:

        result = (
            test_target_temperature(
                target_temp
            )
        )

        results.append(
            result
        )

        print(
            f"Target = {target_temp:>4.1f} °C"
            f"  |  Contrast = "
            f"{result['contrast']:>4.1f} °C"
            f"  →  "
            f"{result['detected']:>2}/"
            f"{result['trials']}"
            f" detected"
            f"  ({result['detection_rate']:>5.1f}%)"
        )

    return results


# ============================================================
# 8. PLOT DETECTION RATE
# ============================================================

def plot_results(
    results
):

    contrasts = [
        result["contrast"]
        for result in results
    ]

    detection_rates = [
        result["detection_rate"]
        for result in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        contrasts,
        detection_rates,
        marker="o"
    )

    # --------------------------------------------------------
    # 80% threshold
    # --------------------------------------------------------

    plt.axhline(
        80,
        linestyle="--",
        label="80% reliable threshold"
    )

    plt.xlabel(
        "Thermal Contrast ΔT (°C)"
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.title(
        "Thermal C Detection Rate vs Thermal Contrast"
    )

    plt.ylim(
        -5,
        105
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 9. SHOW THERMAL IMAGES
# ============================================================

def show_examples(
    results
):

    # --------------------------------------------------------
    # Show selected contrast levels
    # --------------------------------------------------------

    selected_indices = [
        0,
        2,
        4,
        6,
        8,
        9
    ]

    plt.figure(
        figsize=(12, 8)
    )

    for plot_index, result_index in enumerate(
        selected_indices
    ):

        result = results[
            result_index
        ]

        image = result[
            "example_image"
        ]

        target_temp = result[
            "target_temp"
        ]

        contrast = result[
            "contrast"
        ]

        plt.subplot(
            2,
            3,
            plot_index + 1
        )

        plt.imshow(
            image,
            cmap="inferno",
            vmin=0,
            vmax=255
        )

        plt.title(
            f"Target = {target_temp:.1f}°C\n"
            f"ΔT = {contrast:.1f}°C"
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Effect of Thermal Contrast on C Target"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

def print_summary(
    results
):

    print()
    print("=" * 65)
    print("CONTRAST EXPERIMENT SUMMARY")
    print("=" * 65)

    print()

    print(
        f"{'Target °C':<12}"
        f"{'Contrast °C':<15}"
        f"{'Detected':<15}"
        f"{'Rate':<10}"
    )

    print(
        "-" * 52
    )

    for result in results:

        print(
            f"{result['target_temp']:<12.1f}"
            f"{result['contrast']:<15.1f}"
            f"{result['detected']}/"
            f"{result['trials']:<13}"
            f"{result['detection_rate']:.1f}%"
        )


# ============================================================
# 11. FIND 80% RELIABLE CONTRAST
# ============================================================

def find_reliable_contrast(
    results
):
    """
    Find minimum thermal contrast that still
    achieves at least 80% detection.
    """

    reliable_contrasts = []

    for result in results:

        if result["detection_rate"] >= 80:

            reliable_contrasts.append(
                result["contrast"]
            )

    if len(reliable_contrasts) == 0:

        return None

    return min(
        reliable_contrasts
    )


# ============================================================
# 12. MAIN
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
    # Find reliable contrast
    # --------------------------------------------------------

    reliable_contrast = (
        find_reliable_contrast(
            results
        )
    )

    print()
    print("=" * 65)
    print("FINAL RESULT")
    print("=" * 65)

    if reliable_contrast is not None:

        print(
            "Minimum tested thermal contrast "
            f"for 80% reliability: "
            f"ΔT ≥ {reliable_contrast:.1f} °C"
        )

    else:

        print(
            "No tested contrast level achieved "
            "80% reliability."
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Show examples
    # --------------------------------------------------------

    show_examples(
        results
    )


# ============================================================
# 13. RUN
# ============================================================

if __name__ == "__main__":

    main()