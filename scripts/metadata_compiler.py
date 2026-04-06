from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_V3_PATH
from sample_library_cleaner.core.db import DEFAULT_DB_PATH, get_connection
from sample_library_cleaner.data.metadata import rebuild_metadata
from sample_library_cleaner.data.split import assign_splits, load_split_counts


CONFIG_PATH = DEFAULT_CONFIG_V3_PATH
# Switch to `DEFAULT_CONFIG_V3_PATH` once the new label folders are ready.
DB_PATH = DEFAULT_DB_PATH
SAMPLE_ROOTS = [
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/KICKS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNARES",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT CLOSED",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT OPEN",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNAPS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CLAPS",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CYMBALS",
    #"/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/PERCUSSION",
    #"/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/FX",
    #"/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Vocals",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/808s",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Bass",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/BONGO & CONGA",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/CHIMES",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/COWBELL",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/RIMSHOT",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/SHAKER",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/TAMBOURINE",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/TOM",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/TRIANGLE",
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Ambience"
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Textures"
]


def main() -> None:
    # Rebuild metadata first, then assign train/val/test splits on the included rows.
    rebuild_metadata(
        sample_root_paths=SAMPLE_ROOTS,
        config_path=CONFIG_PATH,
        db_path=DB_PATH,
    )

    # Keep split assignment coupled to indexing so one run fully prepares the dataset.
    connection = get_connection(DB_PATH)
    try:
        assign_splits(connection)
        print(load_split_counts(connection))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
