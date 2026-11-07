"""The scoring tests run offline (tests/conftest.py `offline`) and use the reader harness of tests/conftest.py."""
import pytest


@pytest.fixture(autouse=True)
def _offline(offline):
    """No network and dummy credentials for every test here."""
