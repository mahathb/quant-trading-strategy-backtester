"""
Implements the moving average crossover strategy, which is based on the
crossover of short-term and long-term moving averages of the closing price.
"""

from typing import Any

import polars as pl
from quant_trading_strategy_backtester.strategies.base import BaseStrategy
from quant_trading_strategy_backtester.strategy_params import (
    validate_strategy_params,
)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Implements the moving average crossover strategy with an engineered 
    noise-filtering threshold buffer to minimize execution whipsawing.

    Attributes:
        params: A dictionary containing the strategy parameters.
    """

    def __init__(self, params: dict[str, Any]):
        validate_strategy_params("Moving Average Crossover", params)
        super().__init__(params)
        # The number of days for the short-term and long-term moving average.
        self.short_window = int(params["short_window"])
        self.long_window = int(params["long_window"])
        
        # --- B. Mahath Custom Implementation: Fetch Crossover Threshold ---
        # Defaulting to 0.0 if not specified to maintain backward compatibility
        self.crossover_threshold = float(params.get("crossover_threshold", 0.0))

    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Generates trading signals for the given data using Polars expression engines.

        Args:
            data: A DataFrame containing the price data. Must have a 'Close' column.

        Returns:
            A DataFrame containing the generated trading signals.
        """
        if data.is_empty():
            return pl.DataFrame(
                schema=[
                    ("Date", pl.Date),
                    ("Close", pl.Float64),
                    ("short_mavg", pl.Float64),
                    ("long_mavg", pl.Float64),
                    ("signal", pl.Float64),
                    ("position_change", pl.Float64),
                ]
            )

        signals: pl.DataFrame = (  # type: ignore[invalid-assignment]
            data.select([pl.col("Date"), pl.col("Close")])
            .lazy()
            .with_columns(
                [
                    pl.col("Close")
                    .rolling_mean(
                        window_size=self.short_window,
                        min_samples=self.short_window,
                    )
                    .alias("short_mavg"),
                    pl.col("Close")
                    .rolling_mean(
                        window_size=self.long_window,
                        min_samples=self.long_window,
                    )
                    .alias("long_mavg"),
                    pl.lit(0.0).alias("signal"),
                ]
            )
            .with_columns(
                [
                    # --- B. Mahath Custom Implementation: Noise-Filtering Logic ---
                    # A buy signal (1) is only generated when the short-term moving average 
                    # exceeds the long-term moving average by a defined safety buffer percentage.
                    pl.when(pl.col("short_mavg") > (pl.col("long_mavg") * (1 + self.crossover_threshold)))
                    .then(1.0)
                    .otherwise(0.0)
                    .alias("signal")
                ]
            )
            .with_columns(
                [pl.col("signal").diff().fill_null(0.0).alias("position_change")]
            )
            .collect()
        )

        return signals