import pytest
import backend.app as module
import backend.storage as storage


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    # Storage functions read storage.ROOT; app.ROOT is a compatibility re-export.
    monkeypatch.setattr(storage, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    storage.jobs.clear()
