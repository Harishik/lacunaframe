"""Platform accounting and archive protection for the research runners."""
import subprocess
import sys
from pathlib import Path
from research.scripts.compression_pilot import ARCHIVE, host_metadata, peak_rss_mib

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'research/scripts/compression_pilot.py'


def test_peak_rss_units_by_platform():
    assert peak_rss_mib(200 * 1024**2, 'Darwin') == 200  # macOS: bytes
    assert peak_rss_mib(200 * 1024, 'Linux') == 200      # Linux: KiB


def test_host_metadata_has_cpu_and_packages():
    data = host_metadata()
    assert data['cpu_model'] and data['logical_cpus'] and 'onnxruntime' in data['packages']


def test_refuses_to_overwrite_archive():
    result = subprocess.run([sys.executable, str(RUNNER), '--out', str(ARCHIVE)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and 'archived pilot' in result.stderr


def test_refuses_non_empty_trial_directory(tmp_path):
    (tmp_path / 'existing.json').write_text('{}')
    result = subprocess.run([sys.executable, str(RUNNER), '--out', str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0 and 'non-empty' in result.stderr
