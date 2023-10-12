import argparse
import csv
import datetime
import os
from utils.download_data import download_data
from utils.images import create_dataset, ImageType


def run_model(args):
    print(f"Running model with the following parameters:")
    print(f"Model name: {args.model_name}")
    print(f"Epochs: {args.epochs}")


def dataset_enum_type(value):
    try:
        return ImageType.from_string(value)
    except KeyError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid ImageType option")


# Custom argparse type for datetime casting
def datetime_type(value):
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{value} is not a valid datetime in the format YYYY-MM-DD"
        )


def ticker_file_type(value):
    with open(value, "r") as ticker_file:
        rows = csv.DictReader(ticker_file)
        return set(row["Symbol"] for row in rows)


def _create_dataset(args):
    csv_files = [
        os.path.join(args.source, file)
        for file in os.listdir(args.source)
        if file.endswith(".csv")
    ]
    create_dataset(
        args.dataset_name,
        csv_files=csv_files,
        image_type=args.image_type,
    )


def _download_data(args):
    download_data(
        args.ticker_file,
        start=args.start,
        end=args.end,
        directory=args.directory,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Utility for managing datasets and models"
    )

    subparsers = parser.add_subparsers(
        title="subcommands", description="valid subcommands"
    )

    # Subcommand for 'create_dataset'
    parser_dataset = subparsers.add_parser(
        "create_dataset", help="create a new dataset"
    )
    parser_dataset.add_argument(
        "--source", required=True, help="source directory for dataset"
    )
    parser_dataset.add_argument(
        "--dataset-name", required=True, help="name of the dataset to create"
    )
    parser_dataset.add_argument(
        "--image-type",
        type=dataset_enum_type,
        choices=list(ImageType),
        required=True,
        help="image type option",
    )
    parser_dataset.set_defaults(func=_create_dataset)

    # Subcommand for 'run_model'
    parser_model = subparsers.add_parser("run_model", help="run a specified model")
    parser_model.add_argument(
        "--model_name", required=True, help="model to save the name under"
    )
    parser_model.add_argument(
        "--epochs", type=int, default=10, help="number of epochs for training"
    )
    parser_model.set_defaults(func=run_model)

    parser_download = subparsers.add_parser("download_data", help="download stock data")
    parser_download.add_argument(
        "--ticker-file",
        metavar="tickers",
        required=True,
        type=ticker_file_type,
        help="CSV file containing stock-tickers",
    )
    parser_download.add_argument(
        "--start",
        required=True,
        type=datetime_type,
        help="Start datetime in the format YYYY-MM-DD",
    )
    parser_download.add_argument(
        "--end",
        required=True,
        type=datetime_type,
        help="End datetime in the format YYYY-MM-DD",
    )
    parser_download.add_argument(
        "--directory",
        required=True,
        type=str,
        help="Destination directory for downloaded data",
    )
    parser_download.set_defaults(func=_download_data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
