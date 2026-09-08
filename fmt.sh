#!/bin/bash
echo "Running ruff format"
ruff format .
echo "Running ruff check (with auto-fix)"
ruff check --fix .
echo "Done"
