import csv
import os
from pathlib import Path

import yfinance as yf


def _download_data(tickers, *, start, end):
    data = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        ignore_tz=True,
    )
    return data


def _save_data_to_file(day_data, filename):
    Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    with open(filename, "w+") as csv_file:
        rows = [
            row | {"Date": date.strftime("%Y-%m-%d")}
            for date, row in sorted(day_data.items(), key=lambda x: x[0])
        ]
        if rows:
            writer = csv.DictWriter(csv_file, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def _parse_out_data_for_stock(dataframe, ticker):
    sub_frame = dataframe.loc[:, (slice(None), ticker)]
    sub_frame.columns = sub_frame.columns.droplevel(1)
    days_data = sub_frame.to_dict("index")
    return days_data


def download_data(tickers, start, end, directory):
    dataframe = _download_data(tickers, start=start, end=end)
    for ticker in tickers:
        rows = _parse_out_data_for_stock(dataframe, ticker)
        _save_data_to_file(rows, os.path.join(directory, f"{ticker}.csv"))
