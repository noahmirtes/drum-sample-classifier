def main() -> None:
    # Keep the legacy root entrypoint pointed at the packaged v1 runner.
    from sample_library_cleaner.runners.train_v3 import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()
