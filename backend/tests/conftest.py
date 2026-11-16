import pytest
import backend.app as module

@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    module.jobs.clear()
