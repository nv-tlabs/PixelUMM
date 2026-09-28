#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a trusted static DCP export and write its local integrity receipt.

No model weights, metadata, or __SAVE_COMPLETE markers are changed. Existing
receipts are never overwritten. This does not recover an incomplete download
or certify upstream training completion. Use a fully downloaded static export.
"""

import argparse
from pathlib import Path

from train.checkpoint_export import prepare_export_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_export_manifest(args.checkpoint))


if __name__ == "__main__":
    main()
