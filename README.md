# ML-project-2025

# Conf


```bash
conda create -n ml
conda activate ml

# install package
pip3 install --upgrade pip
pip3 install -e .

# install lock
poetry build
poetry install
```

```
hatch run python -c "import kaggle; from kaggle.api.kaggle_api_extended import KaggleApi; api = KaggleApi(); api.authenticate(); api.model_list_cli()"
```

With `kaggle API - download data (train | test | submission)`:

```bash
mkdir data && kaggle competitions download -c playground-series-s5e11 && mv playground-series-s5e11.* data/
```


```bash
unzip data/playground-series-s5e11.zip -d data/
```

## Building an image
```bash
cp Dockerfile_base Dockerfile
docker build -t mlzoomcamp -f Dockerfile .
```

```bash
docker run -it --rm -p 8000:8000 mlzoomcamp
```