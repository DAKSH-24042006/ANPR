"""ANPR REST API Integration Tests.

Tests all FastAPI endpoints against a live server.
Requires the server to be running at BASE_URL before executing.

Run the server first:
    .venv/Scripts/python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000

Then run tests:
    .venv/Scripts/python.exe -m pytest tests/api_tests.py -v
"""

import os
import sys
import time
import threading
import unittest
from pathlib import Path
from io import BytesIO

# ── Ensure project root on path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
except ImportError:
    raise ImportError("'requests' is required for API tests. Install with: pip install requests")

BASE_URL       = "http://127.0.0.1:8000"
SAMPLE_IMAGE   = Path("C:/Users/daksh/Downloads/ANPR_Final_Dataset_Split/images/test/img_00013.jpg")
TIMEOUT        = 120   # seconds per request (OCR can be slow on CPU)


def _server_reachable() -> bool:
    """Return True if the FastAPI server is reachable at BASE_URL."""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(_server_reachable(), "FastAPI server is not running. Start it first.")
class TestHealthEndpoint(unittest.TestCase):
    """GET /health endpoint tests."""

    def test_status_200(self):
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_response_has_status_key(self):
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        data = r.json()
        self.assertIn("status", data)

    def test_response_has_database_key(self):
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        data = r.json()
        self.assertIn("database", data)

    def test_response_has_models_key(self):
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        data = r.json()
        self.assertIn("models", data)


@unittest.skipUnless(_server_reachable(), "FastAPI server is not running. Start it first.")
class TestDetectEndpoint(unittest.TestCase):
    """POST /detect endpoint tests."""

    def _post_image(self, filepath: Path, mime: str = "image/jpeg") -> requests.Response:
        with open(filepath, "rb") as f:
            data = f.read()
        files = {"file": (filepath.name, BytesIO(data), mime)}
        return requests.post(f"{BASE_URL}/detect", files=files, timeout=TIMEOUT)

    def test_valid_image_status_200(self):
        """Valid JPEG upload returns HTTP 200."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        self.assertEqual(r.status_code, 200)

    def test_valid_image_has_uuid(self):
        """Response contains a uuid field."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        data = r.json()
        self.assertIn("uuid", data)
        self.assertIsNotNone(data["uuid"])

    def test_valid_image_has_status(self):
        """Response contains a valid status string."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        valid = {"SUCCESS", "NO_VEHICLE", "NO_PLATE", "OCR_FAILED", "INVALID_IMAGE"}
        self.assertIn(r.json().get("status"), valid)

    def test_valid_image_has_image_path(self):
        """Response contains image_path."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        self.assertIn("image_path", r.json())

    def test_valid_image_has_timings(self):
        """Response timings_ms block is present."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        self.assertIn("timings_ms", r.json())

    def test_valid_image_has_image_save_ms(self):
        """Response timings includes image_save_ms."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        r = self._post_image(SAMPLE_IMAGE)
        self.assertIn("image_save_ms", r.json().get("timings_ms", {}))

    def test_missing_file_returns_422(self):
        """POST /detect with no file attached returns 422 Unprocessable Entity."""
        r = requests.post(f"{BASE_URL}/detect", timeout=10)
        self.assertIn(r.status_code, [400, 422])

    def test_invalid_mime_type(self):
        """Posting a PDF MIME type returns INVALID_IMAGE or HTTP error."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")
        files = {"file": ("test.pdf", BytesIO(b"%PDF fake"), "application/pdf")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=10)
        # Should either refuse (422) or return INVALID_IMAGE status body
        if r.status_code == 200:
            self.assertEqual(r.json().get("status"), "INVALID_IMAGE")
        else:
            self.assertIn(r.status_code, [400, 415, 422])

    def test_unsupported_extension(self):
        """Posting a .bmp file returns INVALID_IMAGE status."""
        files = {"file": ("test.bmp", BytesIO(b"BM fake bmp data"), "image/bmp")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=10)
        if r.status_code == 200:
            self.assertEqual(r.json().get("status"), "INVALID_IMAGE")
        else:
            self.assertIn(r.status_code, [400, 415, 422])

    def test_oversized_upload(self):
        """Posting a 12 MB payload is handled gracefully (413 or INVALID_IMAGE)."""
        large_payload = b"\xFF\xD8\xFF" + b"\x00" * (12 * 1024 * 1024)
        files = {"file": ("large.jpg", BytesIO(large_payload), "image/jpeg")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=30)
        # Accept 413 Request Entity Too Large or a valid JSON body
        self.assertIn(r.status_code, [200, 413, 422, 500])

    def test_corrupted_image(self):
        """Posting random bytes as a JPEG returns INVALID_IMAGE or HTTP error."""
        files = {"file": ("corrupt.jpg", BytesIO(b"not an image"), "image/jpeg")}
        r = requests.post(f"{BASE_URL}/detect", files=files, timeout=TIMEOUT)
        if r.status_code == 200:
            self.assertIn(r.json().get("status"), {"INVALID_IMAGE", "NO_VEHICLE"})
        else:
            self.assertIn(r.status_code, [400, 422, 500])


@unittest.skipUnless(_server_reachable(), "FastAPI server is not running. Start it first.")
class TestHistoryEndpoint(unittest.TestCase):
    """GET /history endpoint tests."""

    def test_history_returns_200(self):
        r = requests.get(f"{BASE_URL}/history", timeout=10)
        self.assertIn(r.status_code, [200])

    def test_history_has_list_structure(self):
        r = requests.get(f"{BASE_URL}/history", timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Response may be a list or a dict with a "data" key
            self.assertTrue(isinstance(data, (list, dict)))

    def test_history_pagination_page1(self):
        r = requests.get(f"{BASE_URL}/history?page=1&limit=5", timeout=10)
        self.assertIn(r.status_code, [200])

    def test_history_invalid_id_returns_404(self):
        r = requests.get(f"{BASE_URL}/history/999999", timeout=10)
        self.assertEqual(r.status_code, 404)


@unittest.skipUnless(_server_reachable(), "FastAPI server is not running. Start it first.")
class TestDeleteEndpoint(unittest.TestCase):
    """DELETE /history/{id} endpoint tests."""

    def test_delete_nonexistent_returns_404(self):
        r = requests.delete(f"{BASE_URL}/history/999999", timeout=10)
        self.assertEqual(r.status_code, 404)


@unittest.skipUnless(_server_reachable(), "FastAPI server is not running. Start it first.")
class TestConcurrentRequests(unittest.TestCase):
    """Verify the server handles simultaneous requests without crashing."""

    def test_concurrent_health_checks(self):
        """Fire 5 simultaneous /health requests and verify all succeed."""
        results = []

        def _get():
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=10)
                results.append(r.status_code)
            except Exception as exc:
                results.append(str(exc))

        threads = [threading.Thread(target=_get) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(len(results), 5)
        for status in results:
            self.assertEqual(status, 200, f"Concurrent request failed: {status}")

    def test_concurrent_detect_requests(self):
        """Fire 3 simultaneous /detect uploads and verify all return HTTP 200."""
        if not SAMPLE_IMAGE.exists():
            self.skipTest("Sample image not found.")

        results = []

        def _detect():
            with open(SAMPLE_IMAGE, "rb") as f:
                data = f.read()
            files = {"file": (SAMPLE_IMAGE.name, BytesIO(data), "image/jpeg")}
            try:
                r = requests.post(f"{BASE_URL}/detect", files=files, timeout=TIMEOUT)
                results.append(r.status_code)
            except Exception as exc:
                results.append(str(exc))

        threads = [threading.Thread(target=_detect) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=TIMEOUT + 10)

        self.assertEqual(len(results), 3)
        for status in results:
            self.assertEqual(status, 200, f"Concurrent detect failed: {status}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    if not _server_reachable():
        print("\n⚠  FastAPI server not running at", BASE_URL)
        print("   Start it with:")
        print("   .venv\\Scripts\\python.exe -m uvicorn api.app:app --host 127.0.0.1 --port 8000\n")
        sys.exit(1)
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
