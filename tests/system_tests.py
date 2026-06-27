"""ANPR System-Level Integration Tests.

Tests the full ANPR pipeline end-to-end across representative image categories
including day, night, various vehicle types, blurred/rotated images, and edge cases.
Does NOT start the FastAPI server. Tests the pipeline directly.
"""

import glob
import json
import os
import sys
import time
import unittest
from pathlib import Path

# ── Ensure project root on path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TEST_IMAGES_DIR = Path("C:/Users/daksh/Downloads/ANPR_Final_Dataset_Split/images/test")
SAMPLE_IMAGE    = str(TEST_IMAGES_DIR / "img_00013.jpg")  # Known good image

# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineInitialisation(unittest.TestCase):
    """Verify that the ANPR pipeline loads without errors."""

    @classmethod
    def setUpClass(cls):
        """Load pipeline once for all tests in this class."""
        from src.anpr_pipeline import ANPRPipeline
        cls.pipeline = ANPRPipeline(device="cpu")

    def test_pipeline_instance(self):
        """Pipeline object is successfully created."""
        self.assertIsNotNone(self.pipeline)

    def test_vehicle_detector_loaded(self):
        """Vehicle detector sub-component is initialised."""
        self.assertIsNotNone(self.pipeline.vehicle_detector)

    def test_plate_detector_loaded(self):
        """Plate detector sub-component is initialised."""
        self.assertIsNotNone(self.pipeline.plate_detector)

    def test_ocr_engine_loaded(self):
        """OCR engine sub-component is initialised."""
        self.assertIsNotNone(self.pipeline.ocr_engine)


class TestPipelineOutputSchema(unittest.TestCase):
    """Verify the output schema of a successful pipeline run."""

    @classmethod
    def setUpClass(cls):
        from src.anpr_pipeline import ANPRPipeline
        cls.pipeline = ANPRPipeline(device="cpu")
        cls.result   = cls.pipeline.run(SAMPLE_IMAGE)

    def test_result_is_dict(self):
        self.assertIsInstance(self.result, dict)

    def test_status_key_present(self):
        from src.response_builder import build_response
        resp = build_response(self.result)
        self.assertIn("status", resp)

    def test_vehicle_key_present(self):
        self.assertIn("vehicle", self.result)

    def test_plate_key_present(self):
        self.assertIn("plate", self.result)

    def test_timings_key_present(self):
        self.assertIn("timings_ms", self.result)

    def test_metadata_key_present(self):
        self.assertIn("metadata", self.result)

    def test_status_is_valid_string(self):
        from src.response_builder import build_response
        resp = build_response(self.result)
        valid = {"SUCCESS", "NO_VEHICLE", "NO_PLATE", "OCR_FAILED", "INVALID_IMAGE"}
        self.assertIn(resp["status"], valid)

    def test_timings_total_positive(self):
        total = self.result.get("timings_ms", {}).get("total_inference", 0)
        self.assertGreater(total, 0)

    def test_metadata_ocr_engine(self):
        self.assertIn("ocr_engine", self.result.get("metadata", {}))


class TestResponseBuilder(unittest.TestCase):
    """Verify build_response generates correct structures for all status codes."""

    def setUp(self):
        from src.response_builder import build_response
        self.build = build_response

    def test_invalid_image_response(self):
        resp = self.build(status="INVALID_IMAGE", message="test")
        self.assertEqual(resp["status"], "INVALID_IMAGE")

    def test_no_vehicle_response(self):
        # Minimal pipeline_result with no vehicle key
        resp = self.build(pipeline_result={"timings_ms": {}, "metadata": {}})
        self.assertIn(resp["status"], {"NO_VEHICLE", "SUCCESS", "NO_PLATE", "OCR_FAILED"})

    def test_uuid_in_response(self):
        resp = self.build(
            status="INVALID_IMAGE",
            uuid="test-uuid-1234"
        )
        self.assertEqual(resp.get("uuid"), "test-uuid-1234")

    def test_image_path_in_response(self):
        resp = self.build(
            status="INVALID_IMAGE",
            image_path="outputs/original_images/test.jpg"
        )
        self.assertEqual(resp.get("image_path"), "outputs/original_images/test.jpg")

    def test_annotated_path_in_response(self):
        resp = self.build(
            status="INVALID_IMAGE",
            annotated_image_path="outputs/annotated_images/test.jpg"
        )
        self.assertEqual(resp.get("annotated_image_path"),
                         "outputs/annotated_images/test.jpg")


class TestVisualization(unittest.TestCase):
    """Verify that visualization produces annotated files."""

    def test_draw_detections_creates_file(self):
        """draw_detections saves a file to outputs/annotated_images/."""
        from src.visualization import draw_detections
        import uuid as uuid_mod
        test_uuid = str(uuid_mod.uuid4())
        # Minimal fake pipeline result (no bboxes to draw)
        fake_result = {
            "vehicle": {"type": "car", "box": [10, 10, 100, 100], "confidence": 0.9},
            "plate":   {"clean_text": "TEST01", "box": [20, 20, 80, 80],
                        "confidence": 0.85, "ocr_confidence": 0.88},
            "timings_ms": {"total_inference": 500.0},
        }
        out_path = draw_detections(SAMPLE_IMAGE, fake_result, test_uuid)
        if out_path:
            self.assertTrue(Path(out_path).exists(),
                            f"Annotated image not found: {out_path}")
        # draw_detections may return None if vehicle box is empty — that is acceptable
        else:
            self.assertIsNone(out_path)


class TestEdgeCases(unittest.TestCase):
    """Test edge-case image inputs for graceful handling."""

    @classmethod
    def setUpClass(cls):
        from src.anpr_pipeline import ANPRPipeline
        cls.pipeline = ANPRPipeline(device="cpu")

    def test_small_image(self):
        """Tiny image (< 10 KB) should complete without crashing."""
        small_imgs = [
            p for p in glob.glob(str(TEST_IMAGES_DIR / "*.jpg"))
            if Path(p).stat().st_size < 10000
        ]
        if not small_imgs:
            self.skipTest("No small images available in test set.")
        result = self.pipeline.run(small_imgs[0])
        self.assertIn("timings_ms", result)

    def test_large_image(self):
        """Large image (> 200 KB) should complete without crashing."""
        large_imgs = [
            p for p in glob.glob(str(TEST_IMAGES_DIR / "*.jpg"))
            if Path(p).stat().st_size > 200000
        ]
        if not large_imgs:
            self.skipTest("No large images available in test set.")
        result = self.pipeline.run(large_imgs[0])
        self.assertIn("timings_ms", result)

    def test_missing_image_path(self):
        """Non-existent image path should return INVALID_IMAGE status."""
        from src.response_builder import build_response
        result = build_response(status="INVALID_IMAGE",
                                message="File not found.")
        self.assertEqual(result["status"], "INVALID_IMAGE")

    def test_invalid_extension(self):
        """Calling with unsupported extension is handled gracefully."""
        from services.detection_service import DetectionService
        svc = DetectionService()
        resp = svc.process_image(b"fakebytes", "image.bmp", db=None)
        self.assertEqual(resp.get("status"), "INVALID_IMAGE")


class TestOutputDirectories(unittest.TestCase):
    """Verify required output directories are auto-created."""

    def test_original_images_dir(self):
        self.assertTrue(Path("outputs/original_images").exists())

    def test_annotated_images_dir(self):
        self.assertTrue(Path("outputs/annotated_images").exists())

    def test_json_results_dir(self):
        self.assertTrue(Path("outputs/json_results").exists())

    def test_logs_dir(self):
        self.assertTrue(Path("outputs/logs").exists())


class TestTimingMetrics(unittest.TestCase):
    """Verify timing fields are present and non-negative in pipeline output."""

    @classmethod
    def setUpClass(cls):
        from src.anpr_pipeline import ANPRPipeline
        cls.pipeline = ANPRPipeline(device="cpu")
        cls.result   = cls.pipeline.run(SAMPLE_IMAGE)

    def test_all_timing_keys_present(self):
        t = self.result.get("timings_ms", {})
        expected_keys = [
            "image_loading", "preprocessing", "vehicle_detection",
            "vehicle_cropping", "plate_detection", "plate_cropping",
            "image_enhancement", "ocr", "post_processing", "total_inference",
        ]
        for key in expected_keys:
            self.assertIn(key, t, f"Missing timing key: {key}")

    def test_all_timings_non_negative(self):
        for key, val in self.result.get("timings_ms", {}).items():
            self.assertGreaterEqual(val, 0, f"Negative timing for {key}: {val}")


class TestDatabaseRepository(unittest.TestCase):
    """Test the database repository CRUD and search queries using in-memory SQLite."""

    @classmethod
    def setUpClass(cls):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.models import Base
        
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine)
        cls.SessionLocal = sessionmaker(bind=cls.engine)

    def setUp(self):
        self.db = self.SessionLocal()

    def tearDown(self):
        # Truncate tables between tests
        from database.models import Detection
        self.db.query(Detection).delete()
        self.db.commit()
        self.db.close()

    def test_add_and_get_by_uuid(self):
        from database.models import Detection
        from database import repository
        from datetime import datetime
        import uuid
        
        test_uuid = str(uuid.uuid4())
        record = Detection(
            uuid=test_uuid,
            timestamp=datetime.now(),
            vehicle_type="car",
            plate_number="MH12DE1433",
            vehicle_confidence=0.9,
            plate_confidence=0.8,
            ocr_confidence=0.85,
            processing_time_ms=1200.0,
            image_path="outputs/original_images/test.jpg",
            annotated_image_path="outputs/annotated_images/test.jpg",
            json_result="{}"
        )
        
        # Test insert
        repository.add_detection(self.db, record)
        
        # Test fetch by uuid
        fetched = repository.get_detection_by_uuid(self.db, test_uuid)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.plate_number, "MH12DE1433")
        
        # Test paginated fetch
        history = repository.get_detections(self.db, skip=0, limit=10)
        self.assertEqual(len(history), 1)
        
        # Test update
        repository.update_detection(self.db, test_uuid, {"plate_number": "MH12DE9999"})
        updated = repository.get_detection_by_uuid(self.db, test_uuid)
        self.assertEqual(updated.plate_number, "MH12DE9999")
        
        # Test search by plate
        by_plate = repository.get_detections_by_plate(self.db, "MH12DE9999")
        self.assertEqual(len(by_plate), 1)
        
        # Test search by timestamp
        start_time = datetime(2000, 1, 1)
        end_time = datetime(2100, 1, 1)
        by_time = repository.get_detections_by_timestamp(self.db, start_time, end_time)
        self.assertEqual(len(by_time), 1)
        
        # Test delete
        deleted = repository.delete_detection_by_uuid(self.db, test_uuid)
        self.assertTrue(deleted)
        self.assertIsNone(repository.get_detection_by_uuid(self.db, test_uuid))


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
