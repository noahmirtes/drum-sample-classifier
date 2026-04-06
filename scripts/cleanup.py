from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_PATH
from sample_library_cleaner.tools.cleanup import print_summary, run_cleanup


PACKS_ROOT_PATH = "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/•NEEDS SORTING•"
CONFIG_PATH = DEFAULT_CONFIG_PATH
DRY_RUN = False
SCAN_RECURSIVELY = False
MOVE_LOG_DIR = Path(__file__).resolve().parent / "results"

TARGET_GROUP_ROOTS: dict[str, str] = {
    "808": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/808s",
    "bass": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Bass",
    "kick": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/KICKS",
    "snare": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNARES",
    "open_hat": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT OPEN",
    "closed_hat": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/HAT CLOSED",
    "cymbal": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CYMBALS",
    "clap": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/CLAPS",
    "snap": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SNAPS",
    "percussion": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/PERCUSSION",
    "fx": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/FX",
    "midi": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/MIDI",
    "vocals": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/MIDI",
    "loops": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Loops",
    "rimshot": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/RIMSHOT",
    "chant": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Vocals/•Chants•",
    "melodies": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Loops/MELODIC LOOPS",
    "tom": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/TOM",
    "bongo & conga": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/BONGO & CONGA",
    "accents": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Textures & Accents",
    "textures": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Textures & Accents",
    "triangle": "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/Drum Samples/SORTED PERCUSSION/TRIANGLE",

}

MANUAL_FOLDER_ALIASES: dict[str, str] = {
    "kicks": "kick",
    "kick": "kick",
    "snares": "snare",
    "snare": "snare",
    "claps": "clap",
    "clap": "clap",
    "snaps": "snap",
    "snap": "snap",
    "closed hats": "closed_hat",
    "closed hat": "closed_hat",
    "hat closed": "closed_hat",
    "hats closed": "closed_hat",
    "ch": "closed_hat",
    "open hats": "open_hat",
    "open hat": "open_hat",
    "hat open": "open_hat",
    "hats open": "open_hat",
    "oh": "open_hat",
    "cymbals": "cymbal",
    "cymbal": "cymbal",
    "crashes": "cymbal",
    "rides": "cymbal",
    "808s": "808",
    "808_s": "808",
    "808 s": "808",
    "808": "808",
    "percs": "percussion",
    "perc": "percussion",
    "percussion": "percussion",
    "fx": "fx",
    "sfx": "fx",
    "fxs": "fx",
    "effects": "fx",
    "midi": "midi",
    "midis": "midi",
    "vox": "vocals",
    "voxs": "vocals",
    "voxes": "vocals",
    "vocals": "vocals",
    "vocal chops": "vocals",
    "countersnare": "snare",
    "countersnares": "snare",
    "counter snare": "snare",
    "counter snares": "snare",
    "loops": "loops",
    "bass": "bass",
    "rim": "rimshot",
    "rims": "rimshot",
    "rimshot": "rimshot",
    "chant": "chant",
    "chants": "chant",
    "melodies": "melodies",
    "toms": "tom",
    "tom": "tom",
    "bongo & conga": "bongo & conga",
    "bongo": "bongo & conga",
    "conga": "bongo & conga",
    "textures": "textures",
    "accents": "accents",
}


def main() -> None:
    # Keep a thin manual runner at repo root while the real logic lives in the package.
    operations, move_log_path = run_cleanup(
        packs_root_path=PACKS_ROOT_PATH,
        target_group_roots=TARGET_GROUP_ROOTS,
        manual_folder_aliases=MANUAL_FOLDER_ALIASES,
        config_path=CONFIG_PATH,
        dry_run=DRY_RUN,
        scan_recursively=SCAN_RECURSIVELY,
        move_log_dir=MOVE_LOG_DIR,
    )
    print_summary(operations, dry_run=DRY_RUN)
    if move_log_path is not None:
        print(f"\nSaved move log: {move_log_path}")


if __name__ == "__main__":
    main()
