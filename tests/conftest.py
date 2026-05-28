import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requires network access")
