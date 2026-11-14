import urllib.error

import pytest

pytest.importorskip("platformdirs")

from waveorder.plugin.samples import download_and_unzip


def test_download_and_unzip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))  # platformdirs' user data directory, under /tmp
    try:
        p1, p2 = download_and_unzip("target")
    except urllib.error.URLError as e:
        pytest.skip(f"sample data server unavailable: {e}")

    assert p1.exists()
    assert p2.exists()
    assert tmp_path in p1.parents
