# adsim, the simulated market.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY adsim ./adsim
RUN pip install --no-cache-dir .
ENTRYPOINT ["python", "-m", "adsim"]
