import openai
import pandas as pd
import typer
import os
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(root_path)

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from typer import Option

from config import ENV_VARIABLES

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main(
        data_path: str = Option(..., help="Path to the data file"),
        collection_name: str = Option(..., help="Name of the collection"),
        batch_size: int = Option(100, help="Batch size for processing embeddings"),
        upsert_batch_size: int = Option(500, help="Batch size for upsert operations")
):
    openai_client = openai.Client(
        api_key=ENV_VARIABLES['OPENAI_API_KEY']
    )

    client = QdrantClient(ENV_VARIABLES['QDRANT_HOST'], port=ENV_VARIABLES['QDRANT_PORT'])

    embedding_model = "text-embedding-3-small"

    train = pd.read_csv(data_path)

    embeddings = []
    for text_batch in batch(list(train['text']), batch_size):
        batch_embeddings = openai_client.embeddings.create(
            input=text_batch,
            model=embedding_model
        ).data
        embeddings.extend([x.embedding for x in batch_embeddings])

    train['embeddings'] = embeddings

    points = [
        PointStruct(
            id=idx,
            vector=row['embeddings'],
            payload={"text": row['text'], "category": row['category']}
        )
        for idx, row in train.iterrows()
    ]

    client.create_collection(
        collection_name,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    )

    for points_batch in batch(points, upsert_batch_size):
        client.upsert(collection_name, points_batch)

if __name__ == "__main__":
    typer.run(main)
