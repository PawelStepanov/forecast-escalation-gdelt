.PHONY: install test run sensitivity docker clean

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

run:
	python -m src.train --config config.yaml

sensitivity:
	python -m src.sensitivity --config config.yaml

docker:
	docker compose up --build

clean:
	rm -rf data/processed/* outputs/*.csv outputs/*.json outputs/figures/*.png
