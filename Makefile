.PHONY: test

build: clean linter
	mkdir -p src/yt_text_extractor && cp yt-text-extractor.py src/yt_text_extractor/__init__.py
	uv venv -q
	uv sync
	uv build

clean:
	rm -rf .python-version dist *.egg* .venv src yt_text_extractor-* *.egg-info

# publish: test
publish:
	UV_PUBLISH_TOKEN=$$(cat .pypi_token) uv publish

githooks:
	git config --local core.hooksPath .githooks

linter: githooks
	uvx isort *.py
	uvx ruff format --line-length 120 *.py
	uvx ruff check

safety:
	uvx safety check -o bare

test: build
	uv pip install -e ".[test]"
	.venv/bin/pytest test
