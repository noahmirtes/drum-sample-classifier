from pathlib import Path

from sample_library_cleaner.core.config import DEFAULT_CONFIG_V2_PATH, DEFAULT_CONFIG_V3_PATH
from sample_library_cleaner.model.infer import run_inference, run_sort_inference
from sample_library_cleaner.runners.train_v2 import DEFAULT_MODEL_V2_PATH
from sample_library_cleaner.runners.train_v3 import DEFAULT_MODEL_V3_PATH


TARGET_PATHS = [
    "/Volumes/Personal Drive/⚭SOUNDS:KITS⚭/•NEEDS SORTING•/ProducerGrind/INCEPTION [Starter Edition]/PG LIMBO - Drum Shots",
]
RUN_SORT = True
TOP_K = 4
MODEL_PATH = DEFAULT_MODEL_V3_PATH
CONFIG_PATH = DEFAULT_CONFIG_V3_PATH
# Switch MODEL_PATH/CONFIG_PATH to the v3 constants once that model is trained.


def main() -> None:
    # Edit the target paths directly when running inference by hand.
    for raw_path in TARGET_PATHS:
        if not raw_path:
            continue

        target_path = Path(raw_path)
        if RUN_SORT:
            run_sort_inference(
                target_path=target_path,
                model_path=MODEL_PATH,
                config_path=CONFIG_PATH,
                top_k=TOP_K,
            )
        else:
            run_inference(
                target_path=target_path,
                model_path=MODEL_PATH,
                config_path=CONFIG_PATH,
                top_k=TOP_K,
            )


if __name__ == "__main__":
    main()
