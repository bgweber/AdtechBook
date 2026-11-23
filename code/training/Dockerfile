# adtrain, the training service the console starts runs on: the click
# model, conversion datasets, and LightGBM models. LightGBM needs the
# OpenMP runtime (libgomp1), which the slim image doesn't have.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY adtrain ./adtrain
RUN pip install --no-cache-dir .
EXPOSE 8090
ENTRYPOINT ["python", "-m", "adtrain"]
CMD ["serve", "--ledger", "/out/ledger", "--models", "/out/models", "--sim-out", "/out/sim"]
