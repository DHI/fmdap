LIB = fmdap

check: lint test

build: check
	python -m build

lint:
	ruff check $(LIB)

format:
	ruff format $(LIB)

test:
	pytest --disable-warnings

coverage: 
	pytest --cov-report html --cov=$(LIB) tests/


