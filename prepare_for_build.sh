#!/bin/bash
# Script to prepare the build environment for FlashAttention.
#
# Example usage:
#   ./prepare_for_build.sh v2.8.3

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <flash_attention_version>"
    echo "Example: $0 v2.8.3"
    exit 1
fi

FLASH_ATTENTION_VERSION=$1

# Apply patches.
patch_dir="${ROOT}/build_scripts/patches/${FLASH_ATTENTION_VERSION}"

# Not all FlashAttention versions need patches.
if [ ! -d "${patch_dir}" ]; then
    echo "Warning: nothing to patch: patches/${FLASH_ATTENTION_VERSION} directory does not exist"
else
    for patch in "${patch_dir}"/*.patch; do
        # Skip if no patch files exist (only .gitkeep)
        if [ -f "${patch}" ]; then
            patch -p1 -d "${ROOT}" -i "${patch}"
        fi
    done
fi
