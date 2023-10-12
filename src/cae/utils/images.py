import csv
from functools import partial
import math
from enum import Enum

import h5py
from tqdm import tqdm
import numpy as np
from utils.stock_history import preprocess_rows


class ImageType(Enum):
    D5 = 5, 32

    def __init__(self, candles, pixel_height):
        self.candles = candles
        self.pixel_height = pixel_height

    @classmethod
    def from_string(cls, name):
        return cls[name.upper()]


def _get_cell(val, *, min, max, image_height):
    return round((image_height - 1) * (val - min) / (max - min))


def _create_moving_average_candle(
    *,
    image_height,
    prior_moving_average,
    current_moving_average,
    next_moving_average,
    high_price,
    low_price,
):
    candle = np.zeros((1, image_height, 3))
    # Exit early if there is no price range
    if (
        high_price - low_price == 0
        or math.isnan(high_price - low_price)
        or math.isnan(current_moving_average)
    ):
        return candle

    get_price_cell = partial(
        _get_cell, min=low_price, max=high_price, image_height=image_height
    )

    # Calculate intermediate moving averages
    if not math.isnan(prior_moving_average):
        prior_delta = (current_moving_average - prior_moving_average) / 3
        candle[0, get_price_cell(current_moving_average - prior_delta), 0] = 1

    candle[0, get_price_cell(current_moving_average), 1] = 1

    if not math.isnan(next_moving_average):
        next_delta = (next_moving_average - current_moving_average) / 3
        candle[0, get_price_cell(current_moving_average + next_delta), 2] = 1
    return candle


def _create_price_candle(*, image_height, row, high_price, low_price):
    candle = np.zeros((1, image_height, 3))
    # Exit early if there is no price range
    if high_price - low_price == 0:
        return candle

    get_price_cell = partial(
        _get_cell, min=low_price, max=high_price, image_height=image_height
    )
    if not math.isnan(row.low) and not math.isnan(row.high):
        candle[
            0,
            get_price_cell(row.low) : get_price_cell(row.high),
            1,
        ] = 1.0
    if not math.isnan(row.open):
        candle[0, get_price_cell(row.open), 0] = 1
    if not math.isnan(row.close):
        candle[0, get_price_cell(row.close), 0] = 1
    return candle


def _create_volume_candle(*, image_height, row, max_volume, min_volume):
    candle = np.zeros((1, image_height, 3))
    if max_volume - min_volume == 0 or math.isnan(max_volume - min_volume):
        return candle

    if not math.isnan(row.volume):
        volume_cell = _get_cell(
            row.volume,
            min=min_volume,
            max=max_volume,
            image_height=image_height,
        )
        candle[0, 0:volume_cell, 2] = 1
    return candle


def _create_candle(
    *,
    image_height,
    row,
    prior_moving_average,
    current_moving_average,
    next_moving_average,
    high_price,
    low_price,
    max_volume,
    min_volume,
):
    return np.vstack(
        (
            _create_price_candle(
                image_height=image_height,
                row=row,
                high_price=high_price,
                low_price=low_price,
            ),
            _create_moving_average_candle(
                image_height=image_height,
                prior_moving_average=prior_moving_average,
                current_moving_average=current_moving_average,
                next_moving_average=next_moving_average,
                high_price=high_price,
                low_price=low_price,
            ),
            _create_volume_candle(
                image_height=image_height,
                row=row,
                max_volume=max_volume,
                min_volume=min_volume,
            ),
        )
    )


def _rows_to_image(relevant_rows, image_height, moving_average_duration):
    # Calculate the image max and min prices
    high_price = max(
        [
            *(r.high for r in relevant_rows if not math.isnan(r.high)),
            *(
                r.moving_averages.get(moving_average_duration, math.nan)
                for r in relevant_rows
                if not math.isnan(
                    r.moving_averages.get(moving_average_duration, math.nan)
                )
            ),
        ],
        default=0,
    )
    low_price = min(
        [
            *(r.low for r in relevant_rows if not math.isnan(r.low)),
            *(
                r.moving_averages.get(moving_average_duration, math.nan)
                for r in relevant_rows
                if not math.isnan(r.moving_averages.get(moving_average_duration))
            ),
        ],
        default=0,
    )

    max_volume = max(
        [r.volume for r in relevant_rows if not math.isnan(r.volume)], default=0
    )
    min_volume = min(
        (r.volume for r in relevant_rows if not math.isnan(r.volume)), default=0
    )

    return np.concatenate(
        list(
            _create_candle(
                image_height=image_height,
                row=row,
                prior_moving_average=(
                    relevant_rows[index - 1].moving_averages.get(
                        moving_average_duration
                    )
                    if index > 0
                    else math.nan
                ),
                current_moving_average=row.moving_averages.get(moving_average_duration),
                next_moving_average=(
                    relevant_rows[index + 1].moving_averages.get(
                        moving_average_duration
                    )
                    if index < len(relevant_rows) - 1
                    else math.nan
                ),
                high_price=high_price,
                low_price=low_price,
                max_volume=max_volume,
                min_volume=min_volume,
            )
            for index, row in enumerate(relevant_rows)
        ),
        axis=2,
    )


def _append_to_hdf5_dataset(h5_file, field, data):
    if not len(data):
        return
    dset = h5_file[field]
    dset.resize(dset.shape[0] + len(data), axis=0)
    dset[-len(data) :] = data


def _generate_images(rows, image_type):
    for start, end in zip(range(0, len(rows)), range(image_type.candles, len(rows))):
        image = _rows_to_image(
            rows[start:end], image_type.pixel_height, image_type.candles
        )
        yield image, rows[end - 1]


def _process_images_and_rows(raw_rows, image_type, moving_average_durations=[]):
    processed_rows = preprocess_rows(
        raw_rows, moving_average_durations=moving_average_durations
    )
    row_image_tupes = list(_generate_images(processed_rows, image_type))
    # Allow for there to be no images
    if not row_image_tupes:
        return [], []
    images, rows = zip(*row_image_tupes)
    return images, rows


def create_dataset(
    dataset_name, csv_files, image_type, quiet=False, compression_rate=4
):
    with h5py.File(f"{dataset_name}.hdf5", "a") as dataset_file:
        # Intialize resizeable datasets, reset any prexisting data
        for field in [
            "images",
            "high",
            "low",
            "open",
            "close",
            "volume",
            "mvg_average",
            "date",
            "ticker",
        ]:
            if field in dataset_file:
                del dataset_file[field]
        dataset_file.create_dataset(
            "images",
            shape=(0, 3, image_type.pixel_height, 3 * image_type.candles),
            maxshape=(None, 3, image_type.pixel_height, 3 * image_type.candles),
            dtype="f",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "high",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "low",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "open",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "close",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "volume",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "mvg_average",
            shape=(0,),
            maxshape=(None,),
            dtype="float32",
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "date",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
            compression="gzip",
            compression_rate=compression_rate,
        )
        dataset_file.create_dataset(
            "ticker",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(),
            compression="gzip",
            compression_rate=compression_rate,
        )

        for filename in tqdm(csv_files, desc="Creating dataset files", disable=quiet):
            with open(filename, "r") as source_file:
                images, rows = _process_images_and_rows(
                    list(csv.DictReader(source_file)),
                    image_type,
                    moving_average_durations=[image_type.candles],
                )
                images = np.stack(images, axis=0)
                _append_to_hdf5_dataset(dataset_file, "images", images)
                _append_to_hdf5_dataset(
                    dataset_file, "high", np.array([row.high for row in rows])
                )
                _append_to_hdf5_dataset(
                    dataset_file, "low", np.array([row.low for row in rows])
                )
                _append_to_hdf5_dataset(
                    dataset_file, "open", np.array([row.open for row in rows])
                )
                _append_to_hdf5_dataset(
                    dataset_file, "close", np.array([row.close for row in rows])
                )
                _append_to_hdf5_dataset(
                    dataset_file, "volume", np.array([row.volume for row in rows])
                )
                _append_to_hdf5_dataset(
                    dataset_file,
                    "mvg_average",
                    np.array(
                        [row.moving_averages.get(image_type.candles) for row in rows]
                    ),
                )
                _append_to_hdf5_dataset(
                    dataset_file,
                    "date",
                    np.array([row.date.isoformat() for row in rows]),
                )
                ticker_name = source_file.name.split("/")[-1].split(".")[0]
                _append_to_hdf5_dataset(
                    dataset_file, "ticker", np.array([ticker_name] * len(rows))
                )
