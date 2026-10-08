"""
Experiment 3:
Noise-only thermal C acuity test.

Purpose:
Measure how thermal noise affects
two-level concentric C detection.

Distance : FIXED
Blur     : OFF
Noise    : VARIABLE
Trials   : 20 per noise level
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

DISTANCE = 0.5

TRIALS_PER_NOISE = 20

# Noise standard deviation in °C.
#
# Example:
#
# sigma = 1.0
#
# means random temperature variation
# with standard deviation of approximately 1°C.

NOISE_LEVELS = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0
]


# ============================================================
# 2. ADD THERMAL NOISE
# ============================================================

def add_thermal_noise(
    temperature,
    noise_std
):
    """
    Add Gaussian thermal noise.

    noise_std:
        Standard deviation of noise in °C.
    """

    # --------------------------------------------------------
    # Generate random noise
    # --------------------------------------------------------

    noise = np.random.normal(
        loc=0.0,
        scale=noise_std,
        size=temperature.shape
    )

    # --------------------------------------------------------
    # Add noise to temperature map
    # --------------------------------------------------------

    noisy_temperature = (
        temperature +
        noise
    )

    # --------------------------------------------------------
    # Keep physically reasonable temperature range
    # --------------------------------------------------------

    noisy_temperature = np.clip(
        noisy_temperature,
        BACKGROUND_TEMP,
        TARGET_TEMP
    )

    return noisy_temperature.astype(
        np.float32
    )


# ============================================================
# 3. RUN C DETECTOR
# ============================================================

def detect_target(
    temperature
):
    """
    Preprocess noisy thermal image
    and detect a two-level concentric C.
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
# 4. RUN ONE NOISE LEVEL
# ============================================================

def test_noise_level(
    original_temperature,
    noise_std
):
    """
    Run multiple random trials for
    one noise level.
    """

    detected_count = 0

    example_image = None
    example_binary = None

    # --------------------------------------------------------
    # Multiple trials
    # --------------------------------------------------------

    for trial in range(
        TRIALS_PER_NOISE
    ):

        # ----------------------------------------------------
        # Add random noise
        # ----------------------------------------------------

        noisy_temperature = (
            add_thermal_noise(
                original_temperature,
                noise_std
            )
        )

        # ----------------------------------------------------
        # Run detector
        # ----------------------------------------------------

        detected, cleaned, contours, result = (
            detect_target(
                noisy_temperature
            )
        )

        # ----------------------------------------------------
        # Count detection
        # ----------------------------------------------------

        if detected:

            detected_count += 1

        # ----------------------------------------------------
        # Save first trial for visualization
        # ----------------------------------------------------

        if trial == 0:

            example_image = (
                noisy_temperature.copy()
            )

            example_binary = (
                cleaned.copy()
            )

    # --------------------------------------------------------
    # Detection rate
    # --------------------------------------------------------

    detection_rate = (
        detected_count /
        TRIALS_PER_NOISE
    ) * 100

    return (
        detected_count,
        detection_rate,
        example_image,
        example_binary
    )


# ============================================================
# 5. RUN COMPLETE EXPERIMENT
# ============================================================

def run_experiment():

    print()
    print("=" * 65)
    print("EXPERIMENT 3 — NOISE ONLY")
    print("=" * 65)

    print()
    print(f"Distance : {DISTANCE} m")
    print("Blur     : OFF")
    print(
        f"Trials   : {TRIALS_PER_NOISE}"
    )
    print()

    # --------------------------------------------------------
    # Create clean target
    # --------------------------------------------------------

    original_temperature = (
        create_two_level_target()
    )

    results = []

    # --------------------------------------------------------
    # Test every noise level
    # --------------------------------------------------------

    for noise_std in NOISE_LEVELS:

        (
            detected_count,
            detection_rate,
            example_image,
            example_binary
        ) = test_noise_level(
            original_temperature,
            noise_std
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({
            "noise": noise_std,
            "detected": detected_count,
            "trials": TRIALS_PER_NOISE,
            "detection_rate": detection_rate,
            "example_image": example_image,
            "example_binary": example_binary
        })

        # ----------------------------------------------------
        # Print status
        # ----------------------------------------------------

        print(
            f"Noise σ = {noise_std:>4.1f} °C"
            f"  →  "
            f"{detected_count:>2}/{TRIALS_PER_NOISE}"
            f" detected"
            f"  ({detection_rate:>5.1f}%)"
        )

    return results


# ============================================================
# 6. PLOT DETECTION RATE
# ============================================================

def plot_results(
    results
):

    noise_levels = [
        item["noise"]
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
        noise_levels,
        detection_rates,
        marker="o"
    )

    # --------------------------------------------------------
    # 80% reliability line
    # --------------------------------------------------------

    plt.axhline(
        80,
        linestyle="--",
        label="80% reliable threshold"
    )

    plt.xlabel(
        "Thermal Noise σ (°C)"
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.title(
        "Thermal C Detection Rate vs Noise"
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
# 7. SHOW NOISE EXAMPLES
# ============================================================

def show_examples(
    results
):

    # --------------------------------------------------------
    # Select representative noise levels
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

        noise_std = result[
            "noise"
        ]

        plt.subplot(
            2,
            3,
            plot_index + 1
        )

        plt.imshow(
            image,
            cmap="inferno",
            vmin=BACKGROUND_TEMP,
            vmax=TARGET_TEMP
        )

        plt.title(
            f"Noise σ = {noise_std:.1f} °C"
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Effect of Thermal Noise on C Target"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

def print_summary(
    results
):

    print()
    print("=" * 65)
    print("NOISE EXPERIMENT SUMMARY")
    print("=" * 65)

    print()

    print(
        f"{'Noise σ (°C)':<15}"
        f"{'Detected':<15}"
        f"{'Detection Rate':<15}"
    )

    print(
        "-" * 45
    )

    for result in results:

        print(
            f"{result['noise']:<15.1f}"
            f"{result['detected']}/"
            f"{result['trials']:<13}"
            f"{result['detection_rate']:.1f}%"
        )


# ============================================================
# 9. FIND 80% RELIABLE NOISE LIMIT
# ============================================================

def find_reliable_noise_limit(
    results
):
    """
    Find the highest noise level that still
    achieves at least 80% detection.
    """

    reliable_levels = []

    for result in results:

        if result["detection_rate"] >= 80:

            reliable_levels.append(
                result["noise"]
            )

    if len(reliable_levels) == 0:

        return None

    return max(
        reliable_levels
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    results = run_experiment()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    # --------------------------------------------------------
    # Reliable noise limit
    # --------------------------------------------------------

    reliable_noise = (
        find_reliable_noise_limit(
            results
        )
    )

    print()
    print("=" * 65)
    print("FINAL RESULT")
    print("=" * 65)

    if reliable_noise is not None:

        print(
            f"80% reliable noise limit:"
            f" σ ≤ {reliable_noise:.1f} °C"
        )

    else:

        print(
            "No tested noise level achieved "
            "80% reliability."
        )

    # --------------------------------------------------------
    # Detection curve
    # --------------------------------------------------------

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Visual examples
    # --------------------------------------------------------

    show_examples(
        results
    )


# ============================================================
# 11. RUN
# ============================================================

if __name__ == "__main__":

    main()