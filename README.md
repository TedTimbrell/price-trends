# price-trends

Originally based on the humorous "(Re-)Imag(in)ing Price Trends", which converts stock data into a graph and then trains a convolutional neural net to look at it and compute a probability of the price going up over some time horizon. This project reimplements the paper in pytorch and aims to extend it to end-to-end deep learning contexts.


<img display="block" src="https://github.com/TedTimbrell/price-trends/assets/15372545/327db33c-670b-4877-a3d5-9c783e413640" width="300px" maxWidth="100%">


>Jiang, Jingwen and Kelly, Bryan T. and Xiu, Dacheng, (Re-)Imag(in)ing Price Trends (December 1, 2020). Chicago Booth Research Paper No. 21-01, Available at SSRN: https://ssrn.com/abstract=3756587 or http://dx.doi.org/10.2139/ssrn.3756587

## What this repo contains
* Downloading historical data from yahoo finance
* Generating images and storing in them in compressed hdf5 files (along with candle data useful for post-hoc labelling)
* Simple prebuilt Pytorch dataset class/pattern that can interface with with the hdf5 files
* The original price prediction model and an in-progress auto-encoder.

## Divergence from (Re-)Imag(in)ing Price Trends
It's worth nothing that our image generation isn't quite the same as the original paper. The original paper segments a single channel image, leaving the top for the price candle and the bottom for a smaller graph of the volume.

The generation method in this module splits the price candles, the moving average, and the volume each into their own channel of the image. The intuition behind this is that I as a human struggled to find the moving average in a large number of cases where it overlapped the high and low for a candle on a certain day. Making volume its own channel is a little more dubious but it will be investigated later.

## How to Use 
### Installation
Built wheel and pypi install aren't available yet.

You can clone the repo and install the packages rather simply with [poetry](https://python-poetry.org/docs/) via 
```poetry install```

### CLI
The CLI will let you run and use most of the repo.
#### Download Historical Data
```
poetry run python cae/cli.py download_data --ticker-file "sp500updated_feb_2023.csv" --start "2000-01-01" --end "2023-10-06" --directory "dry_run"
```
This will trivially download data from yahoo finance using [yfinance](https://github.com/ranaroussi/yfinance) (thanks!).

The ticker file must be a csv with a "Symbol" column that matches the yahoo finance ticker.
#### Create Dataset from Historical Data
```
poetry run python cae/cli.py create_dataset --source="dry_run" --dataset-name="testset_gzip" --image-type=D5
```
The files themselves need to be csv files and require the following named headers representing the candle data:
* High: "High" or "high"
* Low: "Low" or "low"
* Open: "Open" or "open"
* Close: "Close" or "close"
* Volume: "Volume" or "volume"
* Adj Close: "Adj Close" or "adjclose"

Adjusted close is used to normalize each row's candle data so that the close matches the adjusted close. 

## Avenues of Extension
Probabilities are needed for stock trading but the original paper isn't an end-to-end model and only finds trading success by implenenting a strategy on top of the predicted probabilities (selecting the bottom and top 10% of confidences).

Regression via a bayesian neural net seems like a natural next step in order to get both the actual predicted return (what we're really trying to optimize!) while still learning/returning some level of confidence interval. 

The other is to learning an embedding for range of trading data via an auto-encoder. This A) is fun and B) if it works will create an embedding that can be used quickly in other models, time horizons, and contexts. 
