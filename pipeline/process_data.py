import pandas as pd
from typing import Annotated
import typer
import logging

from bert.utils import load_data

def main(
        data_path: Annotated[str, typer.Option(help="Path to the data file")],
        train_output_path: Annotated[str, typer.Option(help="Path to save the output file")],
        test_output_path: Annotated[str, typer.Option(help="Path to save the output file")]
):
    train, test = load_data(data_path)

    train.to_csv(train_output_path, index=False)
    logging.info(f"Output file saved at {train_output_path}")

    test.to_csv(test_output_path, index=False)
    logging.info(f"Output file saved at {test_output_path}")

if __name__ == "__main__":
    typer.run(main)