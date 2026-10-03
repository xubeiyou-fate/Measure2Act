"""Launch the packaged ASCENT evaluation entrypoint."""

from .evaluate import main as evaluate


def main() -> None:
    evaluate()


if __name__ == "__main__":
    main()
