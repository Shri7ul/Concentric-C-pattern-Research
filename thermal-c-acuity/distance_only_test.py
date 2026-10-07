"""
Experiment 1:
Distance-only thermal C acuity test.

Purpose:
Measure the effect of target size reduction alone.

NO optical blur.
NO thermal noise.

Only distance changes the apparent target size.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess

from c_detector import (
    create_two_level_target,
    detect_concentric_pair,

    BACKGROUND_TEMP,
    TARGET_TEMP,

    THRESHOLD,
    KERNEL_SIZE
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Reference distance
# ------------------------------------------------------------

REFERENCE_DISTANCE = 0.5


# ------------------------------------------------------------
# Distances
# ------------------------------------------------------------

# Finer spacing around the transition region
DISTANCES = np.arange(
    0.5,
    1.51,
    0.1
)


# ------------------------------------------------------------
# Number of trials
# ------------------------------------------------------------

# Since there is NO random noise in this experiment,
# repeated trials should produce the same result.
#
# We keep this as 1 because one deterministic test
# is enough.
TRIALS_PER_DISTANCE = 1


# ============================================================
# 2. DISTANCE SIMULATION
# ============================================================

def simulate_distance(
    temperature,
    distance,
    reference_distance=REFERENCE_DISTANCE
):
    """
    Reduce target size according to distance.

    Important:
    This function does NOT add blur.
    This function does NOT add noise.
    """

    height, width = temperature.shape

    # --------------------------------------------------------
    # Scale factor
    # --------------------------------------------------------

    scale = (
        reference_distance /
        distance
    )

    # --------------------------------------------------------
    # Find target
    # --------------------------------------------------------

    target_mask = (
        temperature >
        BACKGROUND_TEMP
    )

    ys, xs = np.where(
        target_mask
    )

    if len(xs) == 0:

        return temperature.copy()

    # --------------------------------------------------------
    # Target bounding box
    # --------------------------------------------------------

    x_min = xs.min()
    x_max = xs.max()

    y_min = ys.min()
    y_max = ys.max()

    target = temperature[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------------
    # New target dimensions
    # --------------------------------------------------------

    new_width = max(
        2,
        int(
            target.shape[1] *
            scale
        )
    )

    new_height = max(
        2,
        int(
            target.shape[0] *
            scale
        )
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized_target = cv2.resize(
        target,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Create background
    # --------------------------------------------------------

    result = np.full(
        (height, width),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Center target
    # --------------------------------------------------------

    center_x = width // 2
    center_y = height // 2

    start_x = (
        center_x -
        new_width // 2
    )

    start_y = (
        center_y -
        new_height // 2
    )

    end_x = (
        start_x +
        new_width
    )

    end_y = (
        start_y +
        new_height
    )

    # --------------------------------------------------------
    # Boundary protection
    # --------------------------------------------------------

    src_x1 = 0
    src_y1 = 0
    src_x2 = new_width
    src_y2 = new_height

    dst_x1 = start_x
    dst_y1 = start_y
    dst_x2 = end_x
    dst_y2 = end_y

    if dst_x1 < 0:

        src_x1 = -dst_x1
        dst_x1 = 0

    if dst_y1 < 0:

        src_y1 = -dst_y1
        dst_y1 = 0

    if dst_x2 > width:

        src_x2 -= (
            dst_x2 -
            width
        )

        dst_x2 = width

    if dst_y2 > height:

        src_y2 -= (
            dst_y2 -
            height
        )

        dst_y2 = height

    # --------------------------------------------------------
    # Paste target
    # --------------------------------------------------------

    result[
        dst_y1:dst_y2,
        dst_x1:dst_x2
    ] = resized_target[
        src_y1:src_y2,
        src_x1:src_x2
    ]

    return result


# ============================================================
# 3. RUN C DETECTOR
# ============================================================

def detect_target(
    temperature
):
    """
    Run preprocessing and concentric C detector.
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
    # Contours
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        cleaned_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # C detector
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
# 4. RUN DISTANCE-ONLY EXPERIMENT
# ============================================================

def run_experiment():

    print()
    print("=" * 65)
    print("EXPERIMENT 1 — DISTANCE ONLY")
    print("=" * 65)

    print()
    print("Blur  : OFF")
    print("Noise : OFF")
    print("Only target size changes.")
    print()

    # --------------------------------------------------------
    # Create original target
    # --------------------------------------------------------

    original_temperature = (
        create_two_level_target()
    )

    results = []

    # --------------------------------------------------------
    # Test every distance
    # --------------------------------------------------------

    for distance in DISTANCES:

        detected_count = 0

        for trial in range(
            TRIALS_PER_DISTANCE
        ):

            # -----------------------------------------------
            # Distance only
            # -----------------------------------------------

            simulated = simulate_distance(
                original_temperature,
                distance
            )

            # -----------------------------------------------
            # Detector
            # -----------------------------------------------

            detected, cleaned, contours, result = (
                detect_target(
                    simulated
                )
            )

            if detected:

                detected_count += 1

        # ----------------------------------------------------
        # Detection rate
        # ----------------------------------------------------

        detection_rate = (
            detected_count /
            TRIALS_PER_DISTANCE
        ) * 100

        results.append({
            "distance": distance,
            "detected": detected,
            "detection_rate": detection_rate
        })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        if detected:

            status = "DETECTED"

        else:

            status = "NOT DETECTED"

        print(
            f"Distance = {distance:.1f} m"
            f"  →  {status}"
        )

    return results


# ============================================================
# 5. PLOT RESULT
# ============================================================

def plot_results(
    results
):

    distances = [
        item["distance"]
        for item in results
    ]

    detection = [
        item["detection_rate"]
        for item in results
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        distances,
        detection,
        marker="o"
    )

    plt.xlabel(
        "Distance (m)"
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.title(
        "Distance-Only Thermal C Detection"
    )

    plt.ylim(
        -5,
        105
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 6. SHOW TARGETS
# ============================================================

def show_examples():

    original_temperature = (
        create_two_level_target()
    )

    example_distances = [
        0.5,
        0.7,
        0.9,
        1.1,
        1.3,
        1.5
    ]

    plt.figure(
        figsize=(12, 8)
    )

    for i, distance in enumerate(
        example_distances
    ):

        simulated = simulate_distance(
            original_temperature,
            distance
        )

        plt.subplot(
            2,
            3,
            i + 1
        )

        plt.imshow(
            simulated,
            cmap="inferno"
        )

        plt.title(
            f"{distance:.1f} m"
        )

        plt.axis(
            "off"
        )

    plt.suptitle(
        "Distance-Only Target Scaling"
    )

    plt.tight_layout()

    plt.show()


# ============================================================
# 7. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    results = run_experiment()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)

    for item in results:

        print(
            f"{item['distance']:.1f} m"
            f" → "
            f"{item['detection_rate']:.0f}%"
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

    show_examples()


# ============================================================
# 8. RUN
# ============================================================

if __name__ == "__main__":

    main()