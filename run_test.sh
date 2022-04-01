#!/bin/bash -e
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -p no:cacheprovider
