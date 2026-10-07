import os
from pathlib import Path

# ----------------------------------------------------------------------
# Project structure definition
# ----------------------------------------------------------------------
# Key   = folder path
# Value = list of files to create inside that folder (empty list = no files)
# ----------------------------------------------------------------------

PROJECT_ROOT = "thermal-c-acuity"

STRUCTURE = {
    # Root level python modules
    ".": [
        "generate_target.py",
        "thermal_model.py",
        "camera_model.py",
        "noise_model.py",
        "preprocessing.py",
        "c_detector.py",
        "acuity_test.py",
    ],

    # Experiments sub-packages
    "experiments": [],
    "experiments/distance":   [],
    "experiments/resolution": [],
    "experiments/noise":      [],
    "experiments/contrast":   [],

    # Results directory
    "results": [],
}

# ----------------------------------------------------------------------
# Optional: docstring header for each generated python file
# ----------------------------------------------------------------------
FILE_HEADERS = {
    "generate_target.py":  '"""Generate thermal test targets (e.g. 4-bar / MRTD style)."""\n',
    "thermal_model.py":    '"""Thermal physics model (temperature, emissivity, radiance)."""\n',
    "camera_model.py":     '"""Camera / sensor model (FOV, IFOV, MTF, detector pitch)."""\n',
    "noise_model.py":      '"""Noise models (NETD, temporal, spatial)."""\n',
    "preprocessing.py":    '"""Preprocessing utilities (NUC, filtering, denoise)."""\n',
    "c_detector.py":       '"""Contrast / C-detector logic for acuity measurement."""\n',
    "acuity_test.py":      '"""Main entry: run thermal acuity test pipeline."""\n',
}


# ----------------------------------------------------------------------
# Creator
# ----------------------------------------------------------------------
def create_structure(root: str = PROJECT_ROOT, structure: dict = STRUCTURE) -> None:
    root_path = Path(root)

    print(f"[+] Creating project root: {root_path.resolve()}")
    root_path.mkdir(parents=True, exist_ok=True)

    for folder, files in structure.items():
        # "." mane root folder
        folder_path = root_path if folder == "." else root_path / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        print(f"    [dir]  {folder_path.relative_to(root_path) or '.'}")

        for fname in files:
            file_path = folder_path / fname
            if file_path.exists():
                print(f"    [skip] {file_path.relative_to(root_path)} (already exists)")
                continue

            # Python file hole header diye create kori
            if fname.endswith(".py"):
                header = FILE_HEADERS.get(fname, f'"""{fname}"""\n')
                file_path.write_text(header, encoding="utf-8")
            else:
                file_path.touch()

            print(f"    [file] {file_path.relative_to(root_path)}")

    print("\n[✓] Structure creation complete!")
    print(f"    -> cd {root}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_structure()