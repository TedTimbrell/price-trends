import pytest
import datetime
from unittest.mock import patch, MagicMock, Mock
from cae.utils.download_data import (
    _save_data_to_file,
    download_data,
)

# Sample data
SAMPLE_TICKERS = ["AAPL"]
SAMPLE_START = datetime.datetime(2020, 1, 1)
SAMPLE_END = datetime.datetime(2020, 1, 2)
SAMPLE_DIRECTORY = "path/to/directory"
SAMPLE_DATAFRAME = MagicMock()
SAMPLE_PARSED_DATA = {SAMPLE_START: {"data": "sample"}}


@pytest.fixture
def mock_yfinance_download(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr("cae.utils.download_data.yf.download", mock)
    return mock


@pytest.fixture
def mock_save_data_to_file(monkeypatch):
    mock = Mock()
    monkeypatch.setattr("cae.utils.download_data._save_data_to_file", mock)
    return mock


def test__save_data_to_file(tmp_path):
    file_path = tmp_path / "test.csv"
    _save_data_to_file(SAMPLE_PARSED_DATA, file_path)
    assert file_path.exists()
    assert (
        file_path.read_text()
        == f"data,Date\nsample,{SAMPLE_START.strftime('%Y-%m-%d')}\n"
    )


def test_download_data(mock_yfinance_download, tmp_path):
    download_data(SAMPLE_TICKERS, SAMPLE_START, SAMPLE_END, tmp_path / SAMPLE_DIRECTORY)
    mock_yfinance_download.assert_called_once()
