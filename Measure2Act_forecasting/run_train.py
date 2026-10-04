"""Launch the packaged aircraft-forecasting training entrypoint."""

from .train import train


def main() -> None:
    train()


if __name__ == "__main__":
    main()
