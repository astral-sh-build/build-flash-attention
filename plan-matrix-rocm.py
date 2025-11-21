# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json

from packaging.version import Version

# Add or remove versions as needed based on flash-attention ROCm compatibility.
# ROCm Flash Attention requires PyTorch 2.2 and later
FLASH_ATTENTION_SUPPORTED_TORCH_VERSIONS = [
    "2.4.1",
    "2.5.1",
    "2.6.0",
    "2.7.1",
    "2.8.0",
    "2.9.0",
]

# ROCm builds are x86_64 only
ARCH_TORCH_PAIRS = {
    "x86_64": ["2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.0"],
}

# Supported Python versions for each PyTorch version.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_PYTHON_SUPPORT = {
    "2.4": ["3.9", "3.10", "3.11", "3.12"],
    "2.5": ["3.9", "3.10", "3.11", "3.12"],
    "2.6": ["3.9", "3.10", "3.11", "3.12"],
    "2.7": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.8": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.9": ["3.10", "3.11", "3.12", "3.13", "3.14"],
}

# ROCm versions to build against for each PyTorch version.
# Only include versions where official PyTorch wheels exist.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
PYTORCH_ROCM_VERSIONS: dict[tuple[str, str], list[str]] = {
    ("2.4", "x86_64"): ["6.1"],  # PyTorch 2.4 officially supports ROCm 6.1
    ("2.5", "x86_64"): ["6.2"],  # PyTorch 2.5 officially supports ROCm 6.2
    ("2.6", "x86_64"): ["6.2"],  # PyTorch 2.6 officially supports ROCm 6.2.4
    ("2.7", "x86_64"): ["6.3"],  # PyTorch 2.7 officially supports ROCm 6.3
    ("2.8", "x86_64"): ["6.4"],  # PyTorch 2.8 officially supports ROCm 6.4
    ("2.9", "x86_64"): ["6.4"],  # PyTorch 2.9 officially supports ROCm 6.4
}

# GPU architectures supported by each ROCm version.
# gfx90a: MI200 series (MI210, MI250, MI250X)
# gfx942: MI300A, MI300X
# gfx950: MI350, MI355 (requires ROCm 7.0+)
ROCM_GPU_ARCHITECTURES: dict[str, list[str]] = {
    "6.1": ["gfx90a", "gfx942"],
    "6.2": ["gfx90a", "gfx942"],
    "6.3": ["gfx90a", "gfx942"],
    "6.4": ["gfx90a", "gfx942"],
}

# The glibc version to use for each PyTorch version, for manylinux builds.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_GLIBC_VERSION: dict[str, str] = {
    "2.4": "2_17",
    "2.5": "2_17",
    "2.6": "2_24",
    "2.7": "2_24",
    "2.8": "2_24",
    "2.9": "2_24",
}

AUDITWHEEL_BLANKET_EXCLUDES = [
    "libc10.so",
    "libc10_hip.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_hip.so",
    "libamdhip64.so",
    "libamdhip64.so.6",
    "libhiprtc.so",
    "libhiprtc.so.6",
    "librocblas.so",
    "librocblas.so.4",
]

AUDITWHEEL_ROCM_VERSION_EXCLUDES = {
    "6": [
        "libhsa-runtime64.so.1",
        "libamd_comgr.so.2",
    ],
}

# Matrix exclusions.
EXCLUSIONS = [
    # No exclusions yet.
]


def main() -> None:
    # Every matrix member is a primary 5-tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.0"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `rocm-version`: the ROCm version as "X.Y", e.g. "6.2"
    # `cxx11-abi`: "TRUE" or "FALSE"
    # `target-arch`: the target architecture, e.g. "x86_64"

    rows = []
    for target_arch, torch_versions in ARCH_TORCH_PAIRS.items():
        for torch_version in torch_versions:
            if torch_version not in FLASH_ATTENTION_SUPPORTED_TORCH_VERSIONS:
                continue

            torch_version_parsed = Version(torch_version)
            torch_x_y = f"{torch_version_parsed.major}.{torch_version_parsed.minor}"
            for python_version in TORCH_PYTHON_SUPPORT[torch_x_y]:
                rocm_versions = PYTORCH_ROCM_VERSIONS[(torch_x_y, target_arch)]
                for rocm_version in rocm_versions:
                    rocm_version_parsed = Version(rocm_version)

                    # The CXX11 ABI became the default in PyTorch 2.7.0, but was also used in
                    # PyTorch 2.6.0 (but _only_ for certain builds).
                    #
                    # See: https://pytorch.org/blog/pytorch2-6/
                    cxx11_abi = torch_version_parsed >= Version("2.7.0")

                    row = {
                        "target-arch": target_arch,
                        "torch-version": str(torch_version_parsed),
                        "python-version": python_version,
                        "rocm-version": rocm_version,
                        "cxx11-abi": "TRUE" if cxx11_abi else "FALSE",
                    }

                    if row not in EXCLUSIONS:
                        rows.append(row)

    rows = rows[:1]

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        # `CI_*` variables: same as the original ones.
        row["CI_ROCM_VERSION"] = row["rocm-version"]
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_ROCM_VERSION`: XY instead of X.Y
        rocm_version = Version(row["rocm-version"])
        row["MATRIX_ROCM_VERSION"] = f"{rocm_version.major}{rocm_version.minor}"

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        torch_version = Version(row["torch-version"])
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # `MANYLINUX_ROCM_VERSION`: X.Y (same as input)
        row["MANYLINUX_ROCM_VERSION"] = f"{rocm_version.major}.{rocm_version.minor}"

        # MANYLINUX_GLIBC_VERSION: the glibc version to use for manylinux builds.
        row["MANYLINUX_GLIBC_VERSION"] = TORCH_GLIBC_VERSION[
            row["MATRIX_TORCH_VERSION"]
        ]

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        rocm_major = str(rocm_version.major)
        auditwheel_excludes = (
            AUDITWHEEL_BLANKET_EXCLUDES
            + AUDITWHEEL_ROCM_VERSION_EXCLUDES.get(rocm_major, [])
        )
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in auditwheel_excludes
        )

        # RUNNER: the GitHub Actions runner to use.
        # ROCm builds run on x86_64
        row["RUNNER"] = "depot-ubuntu-24.04-16"

        # GPU_ARCHS: the GPU architectures to compile for, based on ROCm version
        gpu_archs = ROCM_GPU_ARCHITECTURES.get(row["rocm-version"], ["gfx90a", "gfx942"])
        row["GPU_ARCHS"] = ";".join(gpu_archs)

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
