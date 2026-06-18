# Convenience commands. `make help` lists them.
.PHONY: help install run quick test clean

help:
	@echo "make install   - install python dependencies"
	@echo "make run       - full pipeline (Word2Vec -> bake-off -> test report)"
	@echo "make quick     - fast smoke run on a subset (CI / sanity check)"
	@echo "make test      - run unit tests"
	@echo "make clean     - remove generated models / reports"

install:
	pip install -r requirements.txt

run:
	python run_pipeline.py

quick:
	python run_pipeline.py --quick

test:
	python -m pytest tests/ -q

clean:
	rm -f models/*.keras reports/*.png reports/*.json
