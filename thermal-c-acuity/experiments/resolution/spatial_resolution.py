import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess


# ============================================================
# CONFIGURATION
# ============================================================

WIDTH = 320
HEIGHT = 256

BACKGROUND_TEMP = 25.0
TARGET_TEMP = 40.0

CX = WIDTH // 2
CY = HEIGHT // 2

TRIALS = 20

# Different target sizes
# Smaller radius = fewer pixels = harder to resolve
OUTER_RADII = [
    30,
    29,
    28,
    27,
    26,
    25,
    24,
    23,
    22,
    21,
    20,
]

INNER_RATIO = 0.5

OUTER_WIDTH_RATIO = 0.20
INNER_WIDTH_RATIO = 0.13

GAP_ANGLE = 45


# ============================================================
# DRAW C
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

    y, x = np.ogrid[
        :height,
        :width_image
    ]

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
        np.arctan2(
            y - cy,
            x - cx
        )
    )

    angle = (angle + 360) % 360

    # C opening
    gap = (
        (angle >= 0)
        &
        (angle <= gap_angle)
    )

    c_shape = ring & ~gap

    temperature[c_shape] = target_temp

    return temperature


# ============================================================
# CREATE TARGET
# ============================================================

def create_target(outer_radius):

    temperature = np.full(
        (HEIGHT, WIDTH),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    # ----------------------------
    # Outer C
    # ----------------------------

    outer_width = max(
        2,
        outer_radius * OUTER_WIDTH_RATIO
    )

    temperature = draw_c(
        temperature,
        CX,
        CY,
        radius=outer_radius,
        width=outer_width,
        target_temp=TARGET_TEMP,
        gap_angle=GAP_ANGLE
    )

    # ----------------------------
    # Inner C
    # ----------------------------

    inner_radius = (
        outer_radius * INNER_RATIO
    )

    inner_width = max(
        2,
        outer_radius * INNER_WIDTH_RATIO
    )

    temperature = draw_c(
        temperature,
        CX,
        CY,
        radius=inner_radius,
        width=inner_width,
        target_temp=TARGET_TEMP,
        gap_angle=GAP_ANGLE
    )

    return temperature


# ============================================================
# DETECT CONCENTRIC PAIR
# ============================================================

def detect_concentric_pair(
    contours,
    center_tolerance=5.0,
    radius_ratio_min=1.3
):

    candidates = []

    # --------------------------------------------------------
    # Extract contour information
    # --------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(contour)

        if area <= 0:
            continue

        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(
            contour
        )

        # Moments
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

        candidates.append({
            "contour": contour,
            "center": np.array([
                center_x,
                center_y
            ]),
            "radius": radius,
            "area": area
        })

    # Need at least two contours
    if len(candidates) < 2:
        return None

    # --------------------------------------------------------
    # Compare every contour pair
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

            if smaller_radius == 0:
                continue

            radius_ratio = (
                larger_radius /
                smaller_radius
            )

            # ------------------------------------------------
            # Conditions
            # ------------------------------------------------

            same_center = (
                center_distance <=
                center_tolerance
            )

            different_levels = (
                radius_ratio >=
                radius_ratio_min
            )

            # ------------------------------------------------
            # Detection
            # ------------------------------------------------

            if (
                same_center
                and
                different_levels
            ):

                if (
                    c1["radius"] >
                    c2["radius"]
                ):

                    outer = c1
                    inner = c2

                else:

                    outer = c2
                    inner = c1

                return {
                    "outer": outer,
                    "inner": inner,
                    "center_distance": center_distance,
                    "radius_ratio": radius_ratio
                }

    return None


# ============================================================
# SINGLE TRIAL
# ============================================================

def run_single_trial(outer_radius):

    # Create target
    temperature = create_target(
        outer_radius
    )

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------

    cleaned_image = preprocess(
        temperature,
        min_temp=25.0,
        max_temp=40.0,
        threshold=80,
        kernel_size=3
    )

    # -----------------------------------------
    # Contours
    # -----------------------------------------

    contours, hierarchy = cv2.findContours(
        cleaned_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # -----------------------------------------
    # Detect two-level C
    # -----------------------------------------

    result = detect_concentric_pair(
        contours
    )

    return (
        result is not None,
        temperature,
        cleaned_image
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

results = []

images = []


print()
print("=" * 70)
print("EXPERIMENT 5 — SPATIAL RESOLUTION")
print("=" * 70)

print()
print("Blur  : OFF")
print("Noise : OFF")
print("Contrast : 15 °C")
print("Only target size changes.")
print()


for radius in OUTER_RADII:

    detected_count = 0

    example_image = None

    for trial in range(TRIALS):

        detected, temperature, cleaned = (
            run_single_trial(radius)
        )

        if detected:

            detected_count += 1

        if trial == 0:

            example_image = temperature

    detection_rate = (
        detected_count /
        TRIALS
    ) * 100

    results.append({
        "radius": radius,
        "detected": detected_count,
        "rate": detection_rate
    })

    images.append({
        "radius": radius,
        "image": example_image
    })

    print(
        f"Outer radius = {radius:>3} px"
        f" → "
        f"{detected_count:>2}/{TRIALS}"
        f" detected"
        f" ({detection_rate:>5.1f}%)"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SPATIAL RESOLUTION SUMMARY")
print("=" * 70)

print()

print(
    "Radius (px)   Detected       Detection Rate"
)

print(
    "-" * 50
)

for result in results:

    print(
        f"{result['radius']:>8}"
        f"       "
        f"{result['detected']:>2}/{TRIALS}"
        f"          "
        f"{result['rate']:>5.1f}%"
    )


# ============================================================
# FIND 80% RELIABLE LIMIT
# ============================================================

reliable_radius = None

for result in results:

    if result["rate"] >= 80:

        reliable_radius = result["radius"]


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if reliable_radius is not None:

    print(
        f"80% reliable minimum outer radius:"
        f" {reliable_radius} pixels"
    )

else:

    print(
        "No tested target size reached "
        "80% reliability."
    )


# ============================================================
# PLOT DETECTION RATE
# ============================================================

radii = [
    r["radius"]
    for r in results
]

rates = [
    r["rate"]
    for r in results
]


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    radii,
    rates,
    marker="o"
)

plt.axhline(
    80,
    linestyle="--",
    label="80% reliable threshold"
)

plt.xlabel(
    "Outer C Radius (pixels)"
)

plt.ylabel(
    "Detection Rate (%)"
)

plt.title(
    "Thermal C Detection Rate vs Spatial Size"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# SHOW TARGETS
# ============================================================

fig, axes = plt.subplots(
    2,
    5,
    figsize=(14, 6)
)

axes = axes.flatten()


for ax, item in zip(
    axes,
    images
):

    ax.imshow(
        item["image"],
        cmap="inferno"
    )

    ax.set_title(
        f"{item['radius']} px"
    )

    ax.axis("off")


plt.suptitle(
    "Effect of Spatial Sampling on Thermal C"
)

plt.tight_layout()

plt.show()