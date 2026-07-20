# Inference image for the wall-to-wall / pilot land-cover maps.
#
# Ships the tuned XGBoost model, its unpickling dependency (Tree_Ensemble.py, which defines
# the _LabelDecodingClassifier the pickle references), and vm_predict.py, with every library
# pinned to the training versions. Build small, run on a temporary Compute Engine VM in the
# GCS bucket's region, delete the VM afterwards.
#
#   docker build -t REGION-docker.pkg.dev/PROJECT/REPO/lulc-inference:v1 .
#   docker push  REGION-docker.pkg.dev/PROJECT/REPO/lulc-inference:v1
#
# rasterio ships manylinux wheels with GDAL bundled, so no apt GDAL install is needed -- but
# the wheel still dynamically links a couple of system libs `python:3.11-slim` doesn't
# include: libexpat1 (GDAL's XML parser) and libgomp1 (xgboost's OpenMP runtime). Discovered
# by actually running the built image, not assumed -- see the equivalence test in the plan.
#
# --platform=linux/amd64 is pinned deliberately: Compute Engine VMs are x86_64, but a bare
# `FROM python:3.11-slim` follows whatever architecture the machine running `docker build`
# happens to be. On Apple Silicon that silently produces an arm64 image that pulls and starts
# fine on the VM, then fails every prediction with "exec format error" -- caught here only by
# actually running the image on a real VM, not by the (arm64-native, so falsely reassuring)
# local equivalence test. Pinning the platform makes the image architecture-correct
# regardless of which machine builds it.

FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Unbuffered stdout: without this, `docker logs`/serial output shows nothing until the
# process's internal buffer flushes or the script exits -- fine for the ~20-tile pilot, but
# on the full run (thousands of tiles, hours) it would look silently hung the whole time with
# no way to tell progress from a crash.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libexpat1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so the (slow) pip layer caches across model/script changes.
COPY requirements-vm.txt .
RUN pip install --no-cache-dir -r requirements-vm.txt

# Tree_Ensemble.py MUST be importable to unpickle the model: the pickle stores a reference
# to Tree_Ensemble._LabelDecodingClassifier, not the class itself.
COPY Tree_Ensemble.py vm_predict.py ./

# Bake the model in (9.4 MB) so the image is one immutable artifact: model and library
# versions are locked together and can never drift apart.
COPY Model_Outputs/XGBoost_tuned_model_groupedCV.joblib /app/model/XGBoost_tuned_model_groupedCV.joblib

ENTRYPOINT ["python", "vm_predict.py"]
