#!/usr/bin/bash
pytest --log-cli-level=DEBUG --log-cli-format="%(asctime).19s [%(levelname)s] %(name)s.%(funcName)s (%(filename)s:%(lineno)d):
%(message)s
" --no-print-logs physics_tests.py
