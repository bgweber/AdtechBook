# adtrain, the training service the console starts runs on. Includes
# pandas and pyarrow, to compare predictions with the simulator's true
# click rates.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY adtrain ./adtrain
RUN pip install --no-cache-dir ".[truth]"
EXPOSE 8090
ENTRYPOINT ["python", "-m", "adtrain"]
CMD ["serve", "--ledger", "/out/ledger", "--models", "/out/models", "--sim-out", "/out/sim"]
