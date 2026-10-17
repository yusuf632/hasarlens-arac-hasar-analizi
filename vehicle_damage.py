from datetime import datetime
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import keras
from PIL import Image, UnidentifiedImageError


CLASS_LABELS = (
    "ARKA HASARLI",
    "ARKA HASARSIZ",
    "ÖN HASARLI",
    "ÖN HASARSIZ",
)
IMAGE_SIZE = (160, 160)
MAX_IMAGE_PIXELS = 30_000_000
MODEL_PATH = Path(__file__).resolve().with_name("model_arac_hasar_son.keras")
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    pass


class ModelUnavailableError(RuntimeError):
    pass


def _build_legacy_model() -> keras.Model:
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights=None,
    )
    base_model.trainable = False
    return keras.Sequential(
        [
            base_model,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(0.01),
            ),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(len(CLASS_LABELS), activation="softmax"),
        ]
    )


def load_saved_model() -> keras.Model:
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            f"Model dosyası bulunamadı: {MODEL_PATH.name}"
        )

    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except ValueError:
        try:
            model = _build_legacy_model()
            model.load_weights(MODEL_PATH)
        except (OSError, ValueError) as weights_error:
            raise ModelUnavailableError(
                f"{MODEL_PATH.name} modeli yüklenemedi."
            ) from weights_error
        return model
    except OSError as error:
        raise ModelUnavailableError(
            f"{MODEL_PATH.name} modeli yüklenemedi."
        ) from error


@lru_cache(maxsize=1)
def get_model() -> keras.Model:
    return load_saved_model()


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise ImageValidationError(
                    "Yalnızca JPEG, PNG veya WEBP görselleri destekleniyor."
                )
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ImageValidationError(
                    "Görsel çözünürlüğü çok yüksek. Daha küçük bir görsel deneyin."
                )
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ImageValidationError(
            "Dosya geçerli bir görsel değil veya görsel okunamadı."
        ) from error


def analyze_image(image_bytes: bytes) -> dict[str, object]:
    image = decode_image(image_bytes)
    resized_image = image.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    image_array = keras.utils.img_to_array(resized_image) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)

    predictions = get_model().predict(image_batch, verbose=0)[0]
    probabilities = {
        label: round(float(probability) * 100, 2)
        for label, probability in zip(CLASS_LABELS, predictions)
    }
    class_index = int(np.argmax(predictions))
    label = CLASS_LABELS[class_index]
    is_damaged = "HASARLI" in label

    return {
        "label": label,
        "region": "ÖN" if label.startswith("ÖN") else "ARKA",
        "is_damaged": is_damaged,
        "confidence": probabilities[label],
        "probabilities": probabilities,
        "analyzed_at": datetime.now().astimezone().isoformat(timespec="minutes"),
    }
