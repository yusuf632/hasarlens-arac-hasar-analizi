from io import BytesIO
from pathlib import Path
import unittest

from PIL import Image

from app import app
from vehicle_damage import (
    CLASS_LABELS,
    ImageValidationError,
    decode_image,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color=(25, 80, 120)).save(buffer, format="PNG")
    return buffer.getvalue()


class ImageDecodingTests(unittest.TestCase):
    def test_supported_image_is_converted_to_rgb(self):
        image = decode_image(make_png())

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 6))

    def test_invalid_image_is_rejected(self):
        with self.assertRaises(ImageValidationError):
            decode_image(b"not an image")

    def test_unsupported_format_is_rejected(self):
        buffer = BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, format="BMP")

        with self.assertRaisesRegex(ImageValidationError, "JPEG, PNG veya WEBP"):
            decode_image(buffer.getvalue())


class AnalyzeApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_missing_upload_returns_json_error(self):
        response = self.client.post("/api/analyze")

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_invalid_upload_returns_json_error(self):
        response = self.client.post(
            "/api/analyze",
            data={"image": (BytesIO(b"invalid"), "invalid.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_oversized_upload_returns_json_error(self):
        previous_limit = app.config["MAX_CONTENT_LENGTH"]
        app.config["MAX_CONTENT_LENGTH"] = 128
        self.addCleanup(
            app.config.__setitem__, "MAX_CONTENT_LENGTH", previous_limit
        )

        response = self.client.post(
            "/api/analyze",
            data={"image": (BytesIO(b"x" * 256), "large.jpg")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("error", response.get_json())

    def test_vehicle_photo_returns_four_class_prediction(self):
        image_directory = (
            PROJECT_ROOT / "aracVeriSeti" / "Dataset" / "OnHasarlı"
        )
        image_path = next(
            path for path in sorted(image_directory.iterdir()) if path.is_file()
        )
        response = self.client.post(
            "/api/analyze",
            data={"image": (BytesIO(image_path.read_bytes()), image_path.name)},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(result["label"], CLASS_LABELS)
        self.assertIn(result["region"], ("ÖN", "ARKA"))
        self.assertIsInstance(result["is_damaged"], bool)
        self.assertGreaterEqual(result["confidence"], 0)
        self.assertEqual(set(result["probabilities"]), set(CLASS_LABELS))
        self.assertTrue(result["analyzed_at"])


if __name__ == "__main__":
    unittest.main()
