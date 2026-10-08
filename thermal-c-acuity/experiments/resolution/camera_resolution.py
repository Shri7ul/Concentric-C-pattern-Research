import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 5B — CAMERA RESOLUTION
# ============================================================

BACKGROUND_TEMP = 25.0
TARGET_TEMP = 40.0

DISTANCE = 0.5
TRIALS = 20

# Master / highest resolution
MASTER_WIDTH = 640
MASTER_HEIGHT = 512

# Same physical target
OUTER_RADIUS = 100
INNER_RADIUS = 50

OUTER_WIDTH = 16
INNER_WIDTH = 12

GAP_ANGLE = 45

# Camera resolutions to test
RESOLUTIONS = [
    (640, 512),
    (320, 256),
    (160, 128),
    (80, 64)
]


# ============================================================
# DRAW CONCENTRIC C
# ============================================================

def draw_c(
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

    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    # Ring
    ring = (
        (distance >= radius - width / 2) &
        (distance <= radius + width / 2)
    )

    # Angle
    angle = np.degrees(
        np.arctan2(y - cy, x - cx)
    )

    angle = (angle + 360) % 360

    # Gap
    gap = (
        (angle >= 0) &
        (angle <= gap_angle)
    )

    # C shape
    c_shape = ring & ~gap

    temperature[c_shape] = target_temp

    return temperature


# ============================================================
# CREATE MASTER THERMAL IMAGE
# ============================================================

def create_master_temperature_map():

    temperature = np.full(
        (MASTER_HEIGHT, MASTER_WIDTH),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    cx = MASTER_WIDTH // 2
    cy = MASTER_HEIGHT // 2

    # Outer C
    temperature = draw_c(
        temperature,
        cx,
        cy,
        OUTER_RADIUS,
        OUTER_WIDTH,
        TARGET_TEMP,
        GAP_ANGLE
    )

    # Inner C
    temperature = draw_c(
        temperature,
        cx,
        cy,
        INNER_RADIUS,
        INNER_WIDTH,
        TARGET_TEMP,
        GAP_ANGLE
    )

    return temperature


# ============================================================
# TEMPERATURE → 8-BIT THERMAL IMAGE
# ============================================================

def temperature_to_uint8(
    temperature,
    min_temp=25.0,
    max_temp=40.0
):

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
    ).astype(np.uint8)

    return image


# ============================================================
# SIMULATE CAMERA RESOLUTION
# ============================================================

def simulate_camera_resolution(
    master_image,
    target_width,
    target_height
):

    resized = cv2.resize(
        master_image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    return resized


# ============================================================
# DETECT CONCENTRIC C
# ============================================================

def detect_target(image):

    # Threshold
    _, binary = cv2.threshold(
        image,
        80,
        255,
        cv2.THRESH_BINARY
    )

    # Morphological cleaning
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    # Contours
    contours, hierarchy = cv2.findContours(
        cleaned,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) < 2:
        return False, cleaned

    candidates = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 2:
            continue

        M = cv2.moments(contour)

        if M["m00"] == 0:
            continue

        center_x = (
            M["m10"] /
            M["m00"]
        )

        center_y = (
            M["m01"] /
            M["m00"]
        )

        (_, _), radius = cv2.minEnclosingCircle(
            contour
        )

        candidates.append({
            "center": np.array(
                [center_x, center_y]
            ),
            "radius": radius,
            "area": area
        })

    if len(candidates) < 2:
        return False, cleaned


    # --------------------------------------------------------
    # Check contour pairs
    # --------------------------------------------------------

    for i in range(len(candidates)):

        for j in range(i + 1, len(candidates)):

            c1 = candidates[i]
            c2 = candidates[j]

            # Center distance
            center_distance = np.linalg.norm(
                c1["center"] -
                c2["center"]
            )

            # Radii
            larger_radius = max(
                c1["radius"],
                c2["radius"]
            )

            smaller_radius = min(
                c1["radius"],
                c2["radius"]
            )

            if smaller_radius <= 0:
                continue

            radius_ratio = (
                larger_radius /
                smaller_radius
            )

            # ------------------------------------------------
            # Scale-independent center tolerance
            # ------------------------------------------------

            center_tolerance = (
                0.30 *
                larger_radius
            )

            same_center = (
                center_distance <=
                center_tolerance
            )

            # Two different concentric levels
            different_levels = (
                radius_ratio >= 1.30
            )

            if (
                same_center and
                different_levels
            ):
                return True, cleaned

    return False, cleaned


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 5B — CAMERA RESOLUTION")
    print("=" * 70)

    print(
        "Distance :", DISTANCE, "m"
    )

    print(
        "Noise    : OFF"
    )

    print(
        "Blur     : OFF"
    )

    print(
        "Contrast :",
        TARGET_TEMP - BACKGROUND_TEMP,
        "°C"
    )

    print(
        "Master   :",
        f"{MASTER_WIDTH} × {MASTER_HEIGHT}"
    )

    print()

    # --------------------------------------------------------
    # Create ONE physical target
    # --------------------------------------------------------

    master_temperature = (
        create_master_temperature_map()
    )

    master_image = (
        temperature_to_uint8(
            master_temperature
        )
    )

    results = []

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8)
    )

    axes = axes.flatten()


    # --------------------------------------------------------
    # Test each resolution
    # --------------------------------------------------------

    for index, (width, height) in enumerate(
        RESOLUTIONS
    ):

        detections = 0

        # IMPORTANT:
        # Same physical target is resized
        camera_image = (
            simulate_camera_resolution(
                master_image,
                width,
                height
            )
        )

        # ----------------------------------------------------
        # Repeated trials
        # ----------------------------------------------------

        for trial in range(TRIALS):

            detected, cleaned = detect_target(
                camera_image
            )

            if detected:
                detections += 1

        detection_rate = (
            detections /
            TRIALS *
            100
        )

        results.append({
            "width": width,
            "height": height,
            "detected": detections,
            "rate": detection_rate
        })


        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"Resolution = "
            f"{width:3d} × {height:<3d} "
            f"→ "
            f"{detections:2d}/{TRIALS} "
            f"detected "
            f"({detection_rate:5.1f}%)"
        )


        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        axes[index].imshow(
            camera_image,
            cmap="inferno",
            interpolation="nearest"
        )

        axes[index].set_title(
            f"{width} × {height}\n"
            f"{detection_rate:.1f}% detected"
        )

        axes[index].axis("off")


    plt.suptitle(
        "Effect of Camera Resolution on Thermal C",
        fontsize=14
    )

    plt.tight_layout()

    plt.show()


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("CAMERA RESOLUTION SUMMARY")
    print("=" * 70)

    print(
        f"{'Resolution':<15}"
        f"{'Detected':<15}"
        f"{'Detection Rate':<15}"
    )

    print("-" * 45)

    for result in results:

        resolution = (
            f"{result['width']}×"
            f"{result['height']}"
        )

        detected = (
            f"{result['detected']}/{TRIALS}"
        )

        rate = (
            f"{result['rate']:.1f}%"
        )

        print(
            f"{resolution:<15}"
            f"{detected:<15}"
            f"{rate:<15}"
        )


    # ========================================================
    # 80% RELIABILITY
    # ========================================================

    reliable_resolution = None

    # Start from LOWEST resolution
    # and find the lowest one that
    # reaches 80% reliability.

    for result in reversed(results):

        if result["rate"] >= 80:

            reliable_resolution = (
                f"{result['width']}×"
                f"{result['height']}"
            )

            break


    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if reliable_resolution:

        print(
            "Lowest tested resolution "
            "meeting 80% reliability:",
            reliable_resolution
        )

    else:

        print(
            "No tested resolution "
            "achieved 80% reliability."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()



"""Camera Resolution Experiment:
A fixed thermal concentric-C target was generated at 640×512 resolution and subsequently downsampled to simulate lower-resolution thermal cameras. 
Under constant distance (0.5 m), contrast (15 °C), zero blur, and zero simulated noise, the target was detected in 20/20 trials at 640×512, 320×256, and 160×128 resolutions.
 At 80×64 resolution, detection failed in all 20 trials. Therefore, the lowest tested resolution achieving the predefined 80% reliability criterion was 160×128."""