import numpy as np
import cv2
import matplotlib.pyplot as plt

from preprocessing import preprocess


# ============================================================
# 1. CONFIGURATION
# ============================================================

WIDTH = 320
HEIGHT = 256

BACKGROUND_TEMP = 25.0
TARGET_TEMP = 40.0

# -----------------------------
# Target geometry
# -----------------------------

OUTER_RADIUS = 30
OUTER_WIDTH = 6

INNER_RADIUS = 15
INNER_WIDTH = 4

GAP_ANGLE = 45

# -----------------------------
# Preprocessing
# -----------------------------

THRESHOLD = 80
KERNEL_SIZE = 3

# -----------------------------
# Detection parameters
# -----------------------------

CENTER_TOLERANCE = 5.0
RADIUS_RATIO_MIN = 1.3

MIN_GAP_ANGLE = 20.0
GAP_DIRECTION_TOLERANCE = 15.0

# -----------------------------
# TEST MODE
# -----------------------------
#
# "C"      -> test actual 2-level C
# "CIRCLE" -> negative test using complete circles
#

TEST_TARGET = "C"


# ============================================================
# 2. CREATE TEMPERATURE MAP
# ============================================================

def create_temperature_map():

    temperature = np.full(
        (HEIGHT, WIDTH),
        BACKGROUND_TEMP,
        dtype=np.float32
    )

    cx = WIDTH // 2
    cy = HEIGHT // 2

    return temperature, cx, cy


# ============================================================
# 3. DRAW C-SHAPE
# ============================================================

def draw_c(
    temperature,
    cx,
    cy,
    radius,
    width,
    target_temp,
    gap_angle=GAP_ANGLE
):
    """
    Draw a C-shaped hot region
    into the temperature map.
    """

    height, width_image = temperature.shape

    # --------------------------------------------------------
    # Coordinate grid
    # --------------------------------------------------------

    y, x = np.ogrid[
        :height,
        :width_image
    ]

    # --------------------------------------------------------
    # Distance from center
    # --------------------------------------------------------

    distance = np.sqrt(
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    # --------------------------------------------------------
    # Create ring
    # --------------------------------------------------------

    ring = (
        (distance >= radius - width / 2)
        &
        (distance <= radius + width / 2)
    )

    # --------------------------------------------------------
    # Calculate angle
    # --------------------------------------------------------

    angle = np.degrees(
        np.arctan2(
            y - cy,
            x - cx
        )
    )

    angle = (
        angle + 360
    ) % 360

    # --------------------------------------------------------
    # Create gap
    # --------------------------------------------------------

    gap = (
        (angle >= 0)
        &
        (angle <= gap_angle)
    )

    # --------------------------------------------------------
    # C = ring without gap
    # --------------------------------------------------------

    c_shape = ring & ~gap

    # --------------------------------------------------------
    # Set target temperature
    # --------------------------------------------------------

    temperature[c_shape] = target_temp

    return temperature


# ============================================================
# 4. CREATE TWO-LEVEL C TARGET
# ============================================================

def create_two_level_target():

    temperature, cx, cy = (
        create_temperature_map()
    )

    # --------------------------------------------------------
    # Outer C
    # --------------------------------------------------------

    temperature = draw_c(
        temperature,
        cx,
        cy,
        radius=OUTER_RADIUS,
        width=OUTER_WIDTH,
        target_temp=TARGET_TEMP
    )

    # --------------------------------------------------------
    # Inner C
    # --------------------------------------------------------

    temperature = draw_c(
        temperature,
        cx,
        cy,
        radius=INNER_RADIUS,
        width=INNER_WIDTH,
        target_temp=TARGET_TEMP
    )

    return temperature


# ============================================================
# 5. CREATE TWO COMPLETE CIRCLES
# ============================================================

def create_two_level_circles():
    """
    Negative test.

    Creates two complete concentric circles
    instead of Cs.

    Expected detector result:
        No 2-level C detected.
    """

    temperature, cx, cy = (
        create_temperature_map()
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
        (x - cx) ** 2 +
        (y - cy) ** 2
    )

    # --------------------------------------------------------
    # Outer complete circle
    # --------------------------------------------------------

    outer_circle = (
        (distance >= OUTER_RADIUS - OUTER_WIDTH / 2)
        &
        (distance <= OUTER_RADIUS + OUTER_WIDTH / 2)
    )

    temperature[
        outer_circle
    ] = TARGET_TEMP

    # --------------------------------------------------------
    # Inner complete circle
    # --------------------------------------------------------

    inner_circle = (
        (distance >= INNER_RADIUS - INNER_WIDTH / 2)
        &
        (distance <= INNER_RADIUS + INNER_WIDTH / 2)
    )

    temperature[
        inner_circle
    ] = TARGET_TEMP

    return temperature


# ============================================================
# 6. FIND ANGULAR GAP FROM BINARY IMAGE
# ============================================================

def find_image_angular_gap(
    binary_image,
    center,
    radius,
    ring_tolerance,
    num_angles=360
):
    """
    Examine the actual binary image.

    For every angle, check whether a white pixel
    exists around the expected ring radius.

    This avoids relying only on contour points.
    """

    height, width = binary_image.shape

    cx, cy = center

    # --------------------------------------------------------
    # Store whether each angle is occupied
    # --------------------------------------------------------

    occupied = np.zeros(
        num_angles,
        dtype=bool
    )

    # --------------------------------------------------------
    # Examine every angle
    # --------------------------------------------------------

    for i in range(num_angles):

        angle = (
            360.0
            * i
            / num_angles
        )

        theta = np.radians(
            angle
        )

        found_pixel = False

        # ----------------------------------------------------
        # Search around expected radius
        # ----------------------------------------------------

        radius_start = max(
            1,
            int(
                radius -
                ring_tolerance
            )
        )

        radius_end = int(
            radius +
            ring_tolerance
        )

        for r in range(
            radius_start,
            radius_end + 1
        ):

            x = int(
                round(
                    cx +
                    r * np.cos(theta)
                )
            )

            y = int(
                round(
                    cy +
                    r * np.sin(theta)
                )
            )

            # ------------------------------------------------
            # Boundary check
            # ------------------------------------------------

            if (
                x < 0
                or x >= width
                or y < 0
                or y >= height
            ):
                continue

            # ------------------------------------------------
            # Check pixel
            # ------------------------------------------------

            if binary_image[y, x] > 0:

                found_pixel = True
                break

        occupied[i] = found_pixel

    # ========================================================
    # FIND LONGEST EMPTY ANGULAR REGION
    # ========================================================

    # Duplicate array to handle 0° / 360° wrap
    circular = np.concatenate([
        occupied,
        occupied
    ])

    longest_empty = 0
    current_empty = 0
    longest_end = 0

    for i, is_occupied in enumerate(
        circular
    ):

        if not is_occupied:

            current_empty += 1

            if current_empty > longest_empty:

                longest_empty = current_empty
                longest_end = i

        else:

            current_empty = 0

    # --------------------------------------------------------
    # Limit to one complete revolution
    # --------------------------------------------------------

    longest_empty = min(
        longest_empty,
        num_angles
    )

    # --------------------------------------------------------
    # Convert bins → degrees
    # --------------------------------------------------------

    gap_angle = (
        longest_empty
        * 360.0
        / num_angles
    )

    # --------------------------------------------------------
    # Calculate gap center
    # --------------------------------------------------------

    gap_center_index = (
        longest_end
        - longest_empty / 2
    )

    gap_center_index %= num_angles

    gap_center_angle = (
        gap_center_index
        * 360.0
        / num_angles
    )

    return (
        gap_angle,
        gap_center_angle
    )


# ============================================================
# 7. CHECK WHETHER CONTOUR IS C-SHAPED
# ============================================================

def is_c_shape(
    binary_image,
    center,
    radius,
    ring_tolerance,
    min_gap_angle=MIN_GAP_ANGLE
):
    """
    Determine whether a contour contains
    a sufficiently large angular gap.
    """

    gap_angle, gap_center = (
        find_image_angular_gap(
            binary_image,
            center,
            radius,
            ring_tolerance
        )
    )

    # --------------------------------------------------------
    # C decision
    # --------------------------------------------------------

    is_c = (
        gap_angle >= min_gap_angle
    )

    return {
        "is_c": is_c,
        "gap_angle": gap_angle,
        "gap_center": gap_center
    }


# ============================================================
# 8. DETECT TWO CONCENTRIC Cs
# ============================================================

def detect_concentric_pair(
    binary_image,
    contours,
    center_tolerance=CENTER_TOLERANCE,
    radius_ratio_min=RADIUS_RATIO_MIN,
    min_gap_angle=MIN_GAP_ANGLE,
    gap_direction_tolerance=GAP_DIRECTION_TOLERANCE
):
    """
    Find two nested C-shaped contours
    with approximately the same center.
    """

    candidates = []

    # ========================================================
    # Analyze every contour
    # ========================================================

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore invalid contours
        if area <= 0:
            continue

        # ----------------------------------------------------
        # Minimum enclosing circle
        # ----------------------------------------------------

        (x, y), radius = (
            cv2.minEnclosingCircle(
                contour
            )
        )

        # ----------------------------------------------------
        # Calculate contour center
        # ----------------------------------------------------

        M = cv2.moments(
            contour
        )

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

        center = np.array([
            center_x,
            center_y
        ])

        # ----------------------------------------------------
        # Ring tolerance
        #
        # Slightly larger for bigger rings.
        # ----------------------------------------------------

        ring_tolerance = max(
            2,
            int(
                radius * 0.25
            )
        )

        # ----------------------------------------------------
        # Check C shape
        # ----------------------------------------------------

        c_info = is_c_shape(
            binary_image,
            center,
            radius,
            ring_tolerance,
            min_gap_angle
        )

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        candidates.append({

            "contour": contour,

            "center": center,

            "radius": radius,

            "area": area,

            "is_c": c_info["is_c"],

            "gap_angle": c_info["gap_angle"],

            "gap_center": c_info["gap_center"]
        })

    # ========================================================
    # Need at least two contours
    # ========================================================

    if len(candidates) < 2:
        return None

    # ========================================================
    # Compare every pair
    # ========================================================

    for i in range(
        len(candidates)
    ):

        for j in range(
            i + 1,
            len(candidates)
        ):

            c1 = candidates[i]
            c2 = candidates[j]

            # ------------------------------------------------
            # Both must be C-shaped
            # ------------------------------------------------

            if not c1["is_c"]:
                continue

            if not c2["is_c"]:
                continue

            # ------------------------------------------------
            # Center distance
            # ------------------------------------------------

            center_distance = np.linalg.norm(
                c1["center"] -
                c2["center"]
            )

            # ------------------------------------------------
            # Radius ratio
            # ------------------------------------------------

            larger_radius = max(
                c1["radius"],
                c2["radius"]
            )

            smaller_radius = min(
                c1["radius"],
                c2["radius"]
            )

            radius_ratio = (
                larger_radius /
                smaller_radius
            )

            # ------------------------------------------------
            # Same center?
            # ------------------------------------------------

            same_center = (
                center_distance
                <= center_tolerance
            )

            # ------------------------------------------------
            # Different levels?
            # ------------------------------------------------

            different_levels = (
                radius_ratio
                >= radius_ratio_min
            )

            # ------------------------------------------------
            # Gap direction
            # ------------------------------------------------

            gap_difference = abs(
                c1["gap_center"] -
                c2["gap_center"]
            )

            # Circular angle difference
            gap_difference = min(
                gap_difference,
                360 - gap_difference
            )

            same_gap_direction = (
                gap_difference
                <= gap_direction_tolerance
            )

            # =================================================
            # FINAL PAIR CHECK
            # =================================================

            if (
                same_center
                and
                different_levels
                and
                same_gap_direction
            ):

                # Determine outer / inner
                if (
                    c1["radius"]
                    >
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

                    "center_distance":
                        center_distance,

                    "radius_ratio":
                        radius_ratio,

                    "outer_gap_angle":
                        outer["gap_angle"],

                    "inner_gap_angle":
                        inner["gap_angle"],

                    "outer_gap_direction":
                        outer["gap_center"],

                    "inner_gap_direction":
                        inner["gap_center"],

                    "gap_direction_difference":
                        gap_difference
                }

    return None


# ============================================================
# 9. PRINT CONTOUR INFORMATION
# ============================================================

def print_contour_information(
    contours
):

    print()
    print("=" * 60)
    print("CONTOUR INFORMATION")
    print("=" * 60)

    print(
        "Number of contours:",
        len(contours)
    )

    for i, contour in enumerate(
        contours
    ):

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        area = cv2.contourArea(
            contour
        )

        # ----------------------------------------------------
        # Perimeter
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # ----------------------------------------------------
        # Moments
        # ----------------------------------------------------

        M = cv2.moments(
            contour
        )

        if M["m00"] != 0:

            center_x = (
                M["m10"] /
                M["m00"]
            )

            center_y = (
                M["m01"] /
                M["m00"]
            )

        else:

            center_x = 0
            center_y = 0

        # ----------------------------------------------------
        # Equivalent radius
        # ----------------------------------------------------

        equivalent_radius = np.sqrt(
            area / np.pi
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print()
        print(
            f"Contour {i}"
        )

        print(
            "  Area:",
            area
        )

        print(
            "  Perimeter:",
            perimeter
        )

        print(
            "  Bounding box:",
            x,
            y,
            w,
            h
        )

        print(
            "  Center:",
            center_x,
            center_y
        )

        print(
            "  Equivalent radius:",
            equivalent_radius
        )


# ============================================================
# 10. DRAW DETECTED CONTOURS
# ============================================================

def draw_detected_contours(
    binary_image,
    contours
):

    output = cv2.cvtColor(
        binary_image,
        cv2.COLOR_GRAY2BGR
    )

    cv2.drawContours(
        output,
        contours,
        -1,
        (0, 0, 255),
        1
    )

    return output


# ============================================================
# 11. DISPLAY RESULT
# ============================================================

def display_result(
    output
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )
    )

    plt.title(
        "Detected Contours"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ============================================================
# 12. PRINT FINAL DETECTION RESULT
# ============================================================

def print_detection_result(
    result
):

    print()
    print("=" * 60)
    print("DETECTION RESULT")
    print("=" * 60)

    # --------------------------------------------------------
    # Detection failed
    # --------------------------------------------------------

    if result is None:

        print(
            "❌ No 2-level concentric C detected."
        )

        return

    # --------------------------------------------------------
    # Detection successful
    # --------------------------------------------------------

    print(
        "✅ 2-level concentric C detected!"
    )

    print()

    print(
        "Outer radius:",
        result["outer"]["radius"]
    )

    print(
        "Inner radius:",
        result["inner"]["radius"]
    )

    print(
        "Center distance:",
        result["center_distance"]
    )

    print(
        "Radius ratio:",
        result["radius_ratio"]
    )

    print()

    print(
        "Outer gap angle:",
        result["outer_gap_angle"]
    )

    print(
        "Inner gap angle:",
        result["inner_gap_angle"]
    )

    print(
        "Outer gap direction:",
        result["outer_gap_direction"]
    )

    print(
        "Inner gap direction:",
        result["inner_gap_direction"]
    )

    print(
        "Gap direction difference:",
        result["gap_direction_difference"]
    )


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    # ========================================================
    # CREATE TEST TARGET
    # ========================================================

    if TEST_TARGET == "C":

        print()
        print(
            "TEST TARGET: TWO-LEVEL C"
        )

        temperature = (
            create_two_level_target()
        )

    elif TEST_TARGET == "CIRCLE":

        print()
        print(
            "TEST TARGET: TWO COMPLETE CIRCLES"
        )

        temperature = (
            create_two_level_circles()
        )

    else:

        raise ValueError(
            "TEST_TARGET must be 'C' or 'CIRCLE'"
        )

    # ========================================================
    # PREPROCESS
    # ========================================================

    cleaned_image = preprocess(
        temperature,
        min_temp=BACKGROUND_TEMP,
        max_temp=TARGET_TEMP,
        threshold=THRESHOLD,
        kernel_size=KERNEL_SIZE
    )

    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, hierarchy = cv2.findContours(
        cleaned_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # ========================================================
    # PRINT CONTOUR INFORMATION
    # ========================================================

    print_contour_information(
        contours
    )

    # ========================================================
    # DRAW CONTOURS
    # ========================================================

    output = draw_detected_contours(
        cleaned_image,
        contours
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    display_result(
        output
    )

    # ========================================================
    # DETECT TWO-LEVEL C
    # ========================================================

    result = detect_concentric_pair(
        cleaned_image,
        contours
    )

    # ========================================================
    # PRINT FINAL RESULT
    # ========================================================

    print_detection_result(
        result
    )


# ============================================================
# 14. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()