from __future__ import annotations

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH
from sample_library_cleaner.core.db import DEFAULT_DB_PATH
from sample_library_cleaner.data.metadata import rebuild_metadata


CONFIG_PATH = DEFAULT_CONFIG_PATH
DB_PATH = DEFAULT_DB_PATH
SAMPLE_ROOTS = [
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/KICKS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNARES",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT CLOSED",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT OPEN",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNAPS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CLAPS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CYMBALS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/PERCUSSION",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/808s",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/FX",
]


def main() -> None:
    # Edit the config path and sample roots here when preparing a new training dataset.
    rebuild_metadata(
        sample_root_paths=SAMPLE_ROOTS,
        config_path=CONFIG_PATH,
        db_path=DB_PATH,
    )


if __name__ == "__main__":
    main()
