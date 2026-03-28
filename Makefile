.PHONY: format lint test run-app

format:
	black src app scripts tests

lint:
	ruff check src app scripts tests

test:
	pytest

run-app:
	streamlit run app/streamlit_app.py
