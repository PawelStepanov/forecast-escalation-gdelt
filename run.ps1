$ErrorActionPreference = "Stop"
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.train --config config.yaml
