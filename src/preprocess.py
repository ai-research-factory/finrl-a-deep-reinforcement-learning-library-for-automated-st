"""Preprocess script: download AAPL data, add technical indicators, save processed CSV."""

from data.processor import DataProcessor


def main():
    processor = DataProcessor(data_dir="data")

    # Download AAPL data
    data = processor.download_data(
        tickers=["AAPL"],
        start_date="2009-01-01",
        end_date="2021-12-31",
    )

    df = data["AAPL"]

    # Add technical indicators
    df = processor.add_technical_indicators(df)

    # Forward-fill NaN values (indicators produce NaN for initial windows)
    df = df.ffill()

    # Drop any remaining NaN rows at the very start where ffill can't help
    df = df.dropna()

    # Save processed data
    output_path = processor.processed_dir / "AAPL_processed.csv"
    df.to_csv(output_path)
    print(f"Saved processed data ({len(df)} rows) to {output_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
