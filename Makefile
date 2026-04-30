.PHONY: format lint test run-app

format:
	black src app scripts tests

lint:
	ruff check src app scripts tests

test:
	pytest

run-app:
	python3 scripts/patch_streamlit_connection_message.py
	streamlit run app/streamlit_app.py

fetch-artifacts:
	python scripts/fetch_gdrive_artifacts.py
