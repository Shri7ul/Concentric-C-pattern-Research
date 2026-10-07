"""
Main entry:
Run repeated thermal C acuity tests at different distances.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess

from c_detector import (
    create_two_level_target,
    detect_concentric_pair,

    WIDTH,
    HEIGHT,

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

# At this distance, the target has its original/full size.
REFERENCE_DISTANCE = 0.5


# ------------------------------------------------------------
# Distances to test
# ------------------------------------------------------------

DISTANCES = np.arange(
    0.5,
    4.1,
    0.5
)


# ------------------------------------------------------------
# Repeated trials
# ------------------------------------------------------------

# Each distance will be tested multiple times
# because thermal noise is random.
TRIALS_PER_DISTANCE = 20


# ------------------------------------------------------------
# Camera simulation
# ------------------------------------------------------------

# Base optical blur
BASE_BLUR_SIGMA = 0.5


# Additional blur as distance increases
BLUR_PER_METER = 0.8


# ------------------------------------------------------------
# Thermal noise
# ------------------------------------------------------------

# Standard deviation in Celsius
NOISE_STD = 0.30


# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

# Same seed = reproducible experiment
RANDOM_SEED = 42


# ------------------------------------------------------------
# Reliable detection threshold
# ------------------------------------------------------------

# We consider a distance "reliably detected"
# if at least 80% of trials succeed.
RELIABLE_DETECTION_RATE = 80.0


# ============================================================
# 2. CREATE DISTANCE-SCALED TARGET
# ============================================================

def simulate_distance(
    temperature,
    distance,
    reference_distance=REFERENCE_DISTANCE
):
    """
    Simulate the effect of increasing camera-to-target distance.

    As distance increases, the target occupies fewer pixels.
    """

    height, width = temperature.shape

    # --------------------------------------------------------
    # Calculate scale
    # --------------------------------------------------------

    scale = (
        reference_distance /
        distance
    )

    # --------------------------------------------------------
    # Find target pixels
    # --------------------------------------------------------

    target_mask = (
        temperature >
        BACKGROUND_TEMP
    )

    ys, xs = np.where(
        target_mask
    )

    # If target does not exist
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
    # Calculate new target size
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
    # Resize target
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
    # Target center
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
    # Paste resized target
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
# 3. SIMULATE OPTICAL BLUR
# ============================================================

def simulate_blur(
    temperature,
    distance
):
    """
    Simulate optical/spatial blur.

    Blur increases as distance increases.
    """

    sigma = (
        BASE_BLUR_SIGMA
        +
        BLUR_PER_METER
        *
        (
            distance -
            REFERENCE_DISTANCE
        )
    )

    # Never allow zero/negative sigma
    sigma = max(
        0.1,
        sigma
    )

    blurred = cv2.GaussianBlur(
        temperature,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    return blurred


# ============================================================
# 4. SIMULATE THERMAL NOISE
# ============================================================

def add_thermal_noise(
    temperature,
    noise_std=NOISE_STD,
    rng=None
):
    """
    Add Gaussian thermal measurement noise.
    """

    if rng is None:

        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=temperature.shape
    )

    noisy = (
        temperature +
        noise
    )

    return noisy


# ============================================================
# 5. COMPLETE CAMERA SIMULATION
# ============================================================

def simulate_camera(
    original_temperature,
    distance,
    rng
):
    """
    Complete simulated thermal camera:

        Target
          ↓
        Distance scaling
          ↓
        Optical blur
          ↓
        Thermal noise
    """

    # --------------------------------------------------------
    # Distance scaling
    # --------------------------------------------------------

    distance_image = simulate_distance(
        original_temperature,
        distance
    )

    # --------------------------------------------------------
    # Optical blur
    # --------------------------------------------------------

    blurred_image = simulate_blur(
        distance_image,
        distance
    )

    # --------------------------------------------------------
    # Thermal noise
    # --------------------------------------------------------

    noisy_image = add_thermal_noise(
        blurred_image,
        rng=rng
    )

    return noisy_image


# ============================================================
# 6. RUN PREPROCESSING + DETECTOR
# ============================================================

def detect_target(
    temperature
):
    """
    Run:

        temperature
             ↓
        preprocessing
             ↓
        contours
             ↓
        concentric C detector
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
    # Detect two-level C
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
# 7. RUN REPEATED ACUITY EXPERIMENT
# ============================================================

def run_acuity_test():

    print()
    print("=" * 60)
    print("THERMAL C ACUITY — REPEATED TRIAL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Create original two-level C
    # --------------------------------------------------------

    original_temperature = (
        create_two_level_target()
    )

    # --------------------------------------------------------
    # One RNG for the complete experiment
    #
    # This gives:
    # - different noise every trial
    # - reproducible results between runs
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    results = []

    # --------------------------------------------------------
    # Test every distance
    # --------------------------------------------------------

    for distance in DISTANCES:

        detected_count = 0

        # ----------------------------------------------------
        # Repeat at this distance
        # ----------------------------------------------------

        for trial in range(
            TRIALS_PER_DISTANCE
        ):

            # -----------------------------------------------
            # Simulate camera
            # -----------------------------------------------

            simulated_image = simulate_camera(
                original_temperature,
                distance,
                rng
            )

            # -----------------------------------------------
            # Run detector
            # -----------------------------------------------

            detected, cleaned, contours, result = (
                detect_target(
                    simulated_image
                )
            )

            # -----------------------------------------------
            # Count successful detection
            # -----------------------------------------------

            if detected:

                detected_count += 1

        # ----------------------------------------------------
        # Calculate detection rate
        # ----------------------------------------------------

        detection_rate = (
            detected_count /
            TRIALS_PER_DISTANCE
        ) * 100.0

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        results.append({

            "distance": distance,

            "detected_count":
                detected_count,

            "trials":
                TRIALS_PER_DISTANCE,

            "detection_rate":
                detection_rate

        })

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"Distance = {distance:.1f} m"
            f"  →  "
            f"{detected_count}/"
            f"{TRIALS_PER_DISTANCE}"
            f" detected"
            f"  ({detection_rate:.1f}%)"
        )

    return results


# ============================================================
# 8. PRINT SUMMARY
# ============================================================

def print_summary(
    results
):

    print()
    print("=" * 60)
    print("ACUITY TEST SUMMARY")
    print("=" * 60)

    print()

    print(
        f"{'Distance':<15}"
        f"{'Detected':<15}"
        f"{'Detection Rate':<20}"
    )

    print("-" * 50)

    for item in results:

        print(
            f"{item['distance']:<15.1f}"
            f"{item['detected_count']}/"
            f"{item['trials']:<10}"
            f"{item['detection_rate']:.1f}%"
        )


# ============================================================
# 9. FIND MAXIMUM RELIABLE DISTANCE
# ============================================================

def find_reliable_distance(
    results
):
    """
    Find the maximum distance where the detector
    achieves the required detection rate.
    """

    reliable_distances = [

        item["distance"]

        for item in results

        if item["detection_rate"]
        >= RELIABLE_DETECTION_RATE

    ]

    if len(reliable_distances) == 0:

        return None

    return max(
        reliable_distances
    )


# ============================================================
# 10. PLOT DETECTION RATE
# ============================================================

def plot_results(
    results
):

    distances = [

        item["distance"]

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
        distances,
        detection_rates,
        marker="o"
    )

    # --------------------------------------------------------
    # Reliable detection line
    # --------------------------------------------------------

    plt.axhline(
        RELIABLE_DETECTION_RATE,
        linestyle="--",
        label=(
            f"{RELIABLE_DETECTION_RATE:.0f}% "
            "reliable threshold"
        )
    )

    plt.xlabel(
        "Distance (m)"
    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.title(
        "Thermal C Detection Rate vs Distance"
    )

    plt.ylim(
        0,
        105
    )

    plt.grid(
        True
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# 11. SHOW EXAMPLE IMAGES
# ============================================================

def show_distance_examples(
    original_temperature
):
    """
    Show simulated thermal images
    at several distances.
    """

    example_distances = [

        0.5,
        1.0,
        1.5,
        2.5

    ]

    # Use fixed RNG so example images
    # are reproducible.
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    plt.figure(
        figsize=(12, 8)
    )

    for i, distance in enumerate(
        example_distances
    ):

        # ----------------------------------------------------
        # Simulate camera
        # ----------------------------------------------------

        simulated = simulate_camera(
            original_temperature,
            distance,
            rng
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        plt.subplot(
            2,
            2,
            i + 1
        )

        plt.imshow(
            simulated,
            cmap="inferno"
        )

        plt.title(
            f"Distance = {distance:.1f} m"
        )

        plt.axis(
            "off"
        )

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Run repeated experiment
    # --------------------------------------------------------

    results = run_acuity_test()

    # --------------------------------------------------------
    # Print detailed summary
    # --------------------------------------------------------

    print_summary(
        results
    )

    # --------------------------------------------------------
    # Find reliable distance
    # --------------------------------------------------------

    reliable_distance = (
        find_reliable_distance(
            results
        )
    )

    print()
    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    if reliable_distance is not None:

        print(
            f"Reliable detection distance "
            f"({RELIABLE_DETECTION_RATE:.0f}%): "
            f"{reliable_distance:.1f} m"
        )

    else:

        print(
            "No tested distance achieved "
            f"{RELIABLE_DETECTION_RATE:.0f}% "
            "detection rate."
        )

    # --------------------------------------------------------
    # Plot detection curve
    # --------------------------------------------------------

    plot_results(
        results
    )

    # --------------------------------------------------------
    # Show example images
    # --------------------------------------------------------

    original_temperature = (
        create_two_level_target()
    )

    show_distance_examples(
        original_temperature
    )


# ============================================================
# 13. PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()