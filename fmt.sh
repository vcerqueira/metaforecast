#!/bin/bash
echo "Running fmt"
echo "Running isort"
isort . --profile black
echo "Running black"
black . --line-length 100 --target-version py39 --exclude "/(\.git|\.venv|\.vscode|__pycache__)/"
pylint --rcfile=.pylintrc ./metaforecast
