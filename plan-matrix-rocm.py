# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json

from packaging.version import Version

# Build flash-attention using rocm/pytorch Docker images
# Tag format: rocm{ROCM_VERSION}_ubuntu{UBUNTU_VERSION}_py{PYTHON_VERSION}_pytorch_release_{PYTORCH_VERSION}

# PyTorch versions to build
FLASH_ATTENTION_SUPPORTED_TORCH_VERSIONS = [
    "2.7.1",
    "2.8.0",
    # "2.9.0",  # No rocm/pytorch images available yet
]

# ROCm builds are x86_64 only
ARCH_TORCH_PAIRS = {
    "x86_64": ["2.7.1", "2.8.0"],
}

# Available rocm/pytorch image configurations
# Based on actual available tags from https://hub.docker.com/r/rocm/pytorch
ROCM_PYTORCH_IMAGES: dict[tuple[str, str], list[dict]] = {
    # PyTorch 2.7.1 available configurations
    ("2.7", "x86_64"): [
        {"rocm": "6.4.4", "ubuntu": "22.04", "python": "3.10"},
        {"rocm": "6.4.4", "ubuntu": "24.04", "python": "3.12"},
        {"rocm": "7.1", "ubuntu": "22.04", "python": "3.10"},
        {"rocm": "7.1", "ubuntu": "24.04", "python": "3.12"},
    ],
    # PyTorch 2.8.0 available configurations
    ("2.8", "x86_64"): [
        {"rocm": "7.1", "ubuntu": "22.04", "python": "3.10"},
        {"rocm": "7.1", "ubuntu": "24.04", "python": "3.12"},
    ],
}

# GPU architectures supported by each ROCm version.
# gfx90a: MI200 series (MI210, MI250, MI250X)
# gfx942: MI300A, MI300X
# gfx950: MI350, MI355 (requires ROCm 7.0+)
ROCM_GPU_ARCHITECTURES: dict[str, list[str]] = {
    "6.4": ["gfx90a", "gfx942"],
    "7.0": ["gfx90a", "gfx942", "gfx950"],
    "7.1": ["gfx90a", "gfx942", "gfx950"],
}

# Matrix exclusions
EXCLUSIONS = [
    # No exclusions yet
]


def main() -> None:
    # Every matrix member is a tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.1"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `rocm-version`: the ROCm version as "X.Y.Z", e.g. "6.4.4"
    # `ubuntu-version`: Ubuntu version as "XX.XX", e.g. "22.04"
    # `cxx11-abi`: "TRUE" or "FALSE"
    # `target-arch`: the target architecture, e.g. "x86_64"

    rows = []
    for target_arch, torch_versions in ARCH_TORCH_PAIRS.items():
        for torch_version in torch_versions:
            if torch_version not in FLASH_ATTENTION_SUPPORTED_TORCH_VERSIONS:
                continue

            torch_version_parsed = Version(torch_version)
            torch_x_y = f"{torch_version_parsed.major}.{torch_version_parsed.minor}"

            # Get available image configurations for this PyTorch version
            image_configs = ROCM_PYTORCH_IMAGES.get((torch_x_y, target_arch), [])

            for config in image_configs:
                rocm_version = config["rocm"]
                python_version = config["python"]
                ubuntu_version = config["ubuntu"]

                rocm_version_parsed = Version(rocm_version)

                # The CXX11 ABI became the default in PyTorch 2.7.0
                # See: https://pytorch.org/blog/pytorch2-6/
                cxx11_abi = torch_version_parsed >= Version("2.7.0")

                row = {
                    "target-arch": target_arch,
                    "torch-version": str(torch_version_parsed),
                    "python-version": python_version,
                    "rocm-version": rocm_version,
                    "ubuntu-version": ubuntu_version,
                    "cxx11-abi": "TRUE" if cxx11_abi else "FALSE",
                }

                if row not in EXCLUSIONS:
                    rows.append(row)

    # Transform each row to add various nice-to-have representations of fields
    for row in rows:
        # `CI_*` variables: same as the original ones
        row["CI_ROCM_VERSION"] = row["rocm-version"]
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]
        row["CI_UBUNTU_VERSION"] = row["ubuntu-version"]

        # `MATRIX_ROCM_VERSION`: X.Y (major.minor only)
        rocm_version = Version(row["rocm-version"])
        row["MATRIX_ROCM_VERSION"] = f"{rocm_version.major}.{rocm_version.minor}"

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        torch_version = Version(row["torch-version"])
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # DOCKER_IMAGE: the rocm/pytorch image tag to use
        # Format: rocm{ROCM}_ubuntu{UBUNTU}_py{PYTHON}_pytorch_release_{PYTORCH}
        docker_tag = f"rocm{row['rocm-version']}_ubuntu{row['ubuntu-version']}_py{row['python-version']}_pytorch_release_{row['torch-version']}"
        row["DOCKER_IMAGE"] = f"rocm/pytorch:{docker_tag}"

        # RUNNER: the GitHub Actions runner to use
        # ROCm builds run on x86_64
        row["RUNNER"] = "depot-ubuntu-24.04-16"

        # GPU_ARCHS: the GPU architectures to compile for, based on ROCm version
        rocm_major_minor = row["MATRIX_ROCM_VERSION"]
        gpu_archs = ROCM_GPU_ARCHITECTURES.get(rocm_major_minor, ["gfx90a", "gfx942"])
        row["GPU_ARCHS"] = ";".join(gpu_archs)

    # Limit to first row for initial testing
    print(json.dumps(rows[:1]))


if __name__ == "__main__":
    main()
