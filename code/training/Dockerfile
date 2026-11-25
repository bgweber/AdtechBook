# adtrain, the training service the console starts runs on: the click
# model, conversion datasets, LightGBM models, and the deep models. LightGBM needs the
# OpenMP runtime (libgomp1), which the slim image doesn't have.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# PyTorch for the deep models of Part IV, the CPU build: the exercises run
# on a laptop's CPU. For a GPU, install the CUDA build instead.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
# ONNX and ONNX Runtime, to export the deep model and check the export
# against PyTorch (Chapter 28)
RUN pip install --no-cache-dir onnx onnxruntime
COPY pyproject.toml ./
COPY adtrain ./adtrain
RUN pip install --no-cache-dir .
EXPOSE 8090
ENTRYPOINT ["python", "-m", "adtrain"]
CMD ["serve", "--ledger", "/out/ledger", "--models", "/out/models", "--sim-out", "/out/sim"]
