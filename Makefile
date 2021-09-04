TRAIN_DATA='census_predictions/data/census_income_learn.csv'
EVAL_DATA='census_predictions/data/census_income_test.csv'

clean:
	rm -Rf *.egg-info
	rm -Rf build
	rm -Rf dist
	rm -Rf .pytest_cache
	rm -f .coverage

build: clean
	python3 setup.py sdist

run:
	python3 -m census_predictions.main \
	--train_data $(TRAIN_DATA) \
	--eval_data $(EVAL_DATA)

run_mlflow:
	mlflow run . \
	-P train_data=census_predictions/data/census_income_learn.csv \
	-P eval_data=census_predictions/data/census_income_test.csv
