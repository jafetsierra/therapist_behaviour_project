# Use Python 3.10
FROM huggingface/transformers-pytorch-cpu

RUN apt-get update && apt-get -y install gcc g++ && apt-get install -y \
    python3-dev \
    build-essential \
    curl

WORKDIR /program

ENV POETRY_VERSION=1.3.2
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="${PATH}:/root/.local/bin"
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock* /program/

RUN poetry install --no-interaction --no-ansi --without=dev

ENV PYTHONPATH "${PYTHONPATH}:${PWD}"
# disable python output buffering
ENV PYTHONUNBUFFERED 1

# TODO do not copy all files, only the necessary ones
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]