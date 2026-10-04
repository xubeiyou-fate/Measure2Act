"""Launch the packaged Measure2Act forecasting evaluation entrypoint."""

from .evaluate import main as evaluate


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
