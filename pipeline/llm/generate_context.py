import pandas as pd
from typing import Annotated
import typer
import logging

def main(
        data_path: Annotated[str, typer.Option(help="Path to the data file")],
        output_path: Annotated[str, typer.Option(help="Path to save the output file")]
):
    columns_to_join = ['text','category']

    data = pd.read_csv(data_path)

    data['joined'] = data[columns_to_join].apply(lambda x: '->'.join(x.dropna().astype(str)), axis=1)

    combinen_text = '\n'.join(data['joined'])

    with open(output_path, 'w') as f:
        f.write(combinen_text)
    
    logging.info(f"Output file saved at {output_path}")


if __name__ == "__main__":
    typer.run(main)