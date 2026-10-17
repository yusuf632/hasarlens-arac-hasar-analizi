from __future__ import annotations

import hashlib
import os
import random
import sys
from collections import Counter
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from vehicle_damage import (
    CLASS_LABELS,
    IMAGE_SIZE,
    MODEL_PATH,
    load_saved_model,
)


DATASET_PATH = Path(__file__).resolve().parent / "aracVeriSeti" / "Dataset"
CANDIDATE_PATH = MODEL_PATH.with_name("model_arac_hasar_candidate.keras")
CLASS_DIRECTORIES = (
    "ArkaHasarlı",
    "ArkaHasarsız",
    "OnHasarlı",
    "OnHasarsız",
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RANDOM_SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 12
FINE_TUNE_LAYERS = 30


class ImageBatchDataset(keras.utils.PyDataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        *,
        training: bool,
        seed: int,
    ) -> None:
        super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
        self.samples = samples
        self.training = training
        self.random = random.Random(seed)
        self.order = list(range(len(samples)))
        self.on_epoch_end()

    def __len__(self) -> int:
        return (len(self.samples) + BATCH_SIZE - 1) // BATCH_SIZE

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.order[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        images: list[np.ndarray] = []
        labels: list[int] = []

        for sample_index in batch_indices:
            path, label = self.samples[sample_index]
            with Image.open(path) as image:
                resized = image.convert("RGB").resize(
                    IMAGE_SIZE, Image.Resampling.NEAREST
                )
                images.append(np.asarray(resized, dtype=np.float32) / 255.0)
            labels.append(label)

        return np.stack(images), np.asarray(labels, dtype=np.int32)

    def on_epoch_end(self) -> None:
        if self.training:
            self.random.shuffle(self.order)


def collect_samples() -> list[list[Path]]:
    samples_by_class: list[list[Path]] = []
    hash_to_class: dict[str, int] = {}

    for class_index, directory_name in enumerate(CLASS_DIRECTORIES):
        directory = DATASET_PATH / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Eğitim sınıfı bulunamadı: {directory}")

        unique_paths: list[Path] = []
        class_hashes: set[str] = set()
        paths = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"Eğitim sınıfında görsel bulunamadı: {directory}")

        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous_class = hash_to_class.get(digest)
            if previous_class is not None and previous_class != class_index:
                raise ValueError(
                    f"Aynı görsel farklı sınıflarda etiketlenmiş: {path.name}"
                )
            hash_to_class[digest] = class_index
            if digest not in class_hashes:
                class_hashes.add(digest)
                unique_paths.append(path)

        samples_by_class.append(unique_paths)
        print(
            f"{CLASS_LABELS[class_index]}: {len(unique_paths)} tekil görsel "
            f"({len(paths) - len(unique_paths)} yinelenen görsel çıkarıldı)"
        )

    return samples_by_class


def split_samples(
    samples_by_class: list[list[Path]],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    training: list[tuple[Path, int]] = []
    validation: list[tuple[Path, int]] = []
    test: list[tuple[Path, int]] = []

    for class_index, paths in enumerate(samples_by_class):
        shuffled_paths = paths.copy()
        random.Random(RANDOM_SEED + class_index).shuffle(shuffled_paths)
        train_end = int(len(shuffled_paths) * 0.70)
        validation_end = int(len(shuffled_paths) * 0.85)
        if train_end == 0 or validation_end == train_end:
            raise ValueError(
                f"{CLASS_LABELS[class_index]} sınıfında eğitim için yeterli "
                "görsel yok."
            )

        training.extend((path, class_index) for path in shuffled_paths[:train_end])
        validation.extend(
            (path, class_index)
            for path in shuffled_paths[train_end:validation_end]
        )
        test.extend(
            (path, class_index) for path in shuffled_paths[validation_end:]
        )

    return training, validation, test


def build_fine_tuning_model() -> keras.Model:
    source_model = load_saved_model()
    backbone = next(
        layer for layer in source_model.layers if isinstance(layer, keras.Model)
    )
    head_layers = [
        layer
        for layer in source_model.layers
        if layer is not backbone and not isinstance(layer, keras.layers.InputLayer)
    ]

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    augmented = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.04),
            keras.layers.RandomZoom(0.08),
            keras.layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )(inputs)
    outputs = backbone(augmented, training=False)
    for layer in head_layers:
        outputs = layer(outputs)

    model = keras.Model(inputs, outputs, name="vehicle_damage_classifier")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-FINE_TUNE_LAYERS:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def predict_labels(
    model: keras.Model, samples: list[tuple[Path, int]]
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[int] = []
    labels: list[int] = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch = samples[start : start + BATCH_SIZE]
        images: list[np.ndarray] = []
        for path, label in batch:
            with Image.open(path) as image:
                resized = image.convert("RGB").resize(
                    IMAGE_SIZE, Image.Resampling.NEAREST
                )
                images.append(np.asarray(resized, dtype=np.float32) / 255.0)
            labels.append(label)

        probabilities = model.predict(np.stack(images), verbose=0)
        predictions.extend(np.argmax(probabilities, axis=1).tolist())

    return np.asarray(labels), np.asarray(predictions)


def calculate_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    class_f1_scores: list[float] = []
    for class_index in range(len(CLASS_LABELS)):
        true_positive = int(
            np.sum((labels == class_index) & (predictions == class_index))
        )
        false_positive = int(
            np.sum((labels != class_index) & (predictions == class_index))
        )
        false_negative = int(
            np.sum((labels == class_index) & (predictions != class_index))
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1_scores.append(
            2 * true_positive / denominator if denominator else 0.0
        )

    accuracy = float(np.mean(labels == predictions))
    macro_f1 = float(np.mean(class_f1_scores))
    return accuracy, macro_f1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Başlangıç modeli bulunamadı: {MODEL_PATH.name}"
        )

    keras.utils.set_random_seed(RANDOM_SEED)
    training_by_class = collect_samples()
    training, validation, test = split_samples(training_by_class)
    print(
        f"Eğitim: {len(training)}, doğrulama: {len(validation)}, "
        f"bağımsız test: {len(test)} görsel"
    )

    training_counts = Counter(label for _, label in training)
    class_weights = {
        class_index: len(training) / (len(CLASS_LABELS) * count)
        for class_index, count in training_counts.items()
    }
    training_data = ImageBatchDataset(
        training, training=True, seed=RANDOM_SEED
    )
    validation_data = ImageBatchDataset(
        validation, training=False, seed=RANDOM_SEED
    )

    baseline_model = load_saved_model()
    baseline_labels, baseline_predictions = predict_labels(baseline_model, test)
    baseline_accuracy, baseline_macro_f1 = calculate_metrics(
        baseline_labels, baseline_predictions
    )
    print(
        f"Eski model test sonuçları: doğruluk %{baseline_accuracy * 100:.2f}, "
        f"makro F1 %{baseline_macro_f1 * 100:.2f}"
    )

    model = build_fine_tuning_model()
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            CANDIDATE_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        ),
    ]
    model.fit(
        training_data,
        validation_data=validation_data,
        epochs=MAX_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    candidate_model = keras.models.load_model(CANDIDATE_PATH, compile=False)
    candidate_labels, candidate_predictions = predict_labels(candidate_model, test)
    candidate_accuracy, candidate_macro_f1 = calculate_metrics(
        candidate_labels, candidate_predictions
    )
    print(
        f"Yeni model test sonuçları: doğruluk %{candidate_accuracy * 100:.2f}, "
        f"makro F1 %{candidate_macro_f1 * 100:.2f}"
    )

    if (
        candidate_macro_f1 > baseline_macro_f1
        and candidate_accuracy >= baseline_accuracy
    ):
        os.replace(CANDIDATE_PATH, MODEL_PATH)
        print("Yeni model her iki ölçütte de daha iyi; etkin model güncellendi.")
    else:
        CANDIDATE_PATH.unlink()
        print(
            "Yeni model bağımsız testte mevcut modeli geçemedi; "
            "etkin model korunuyor."
        )


if __name__ == "__main__":
    main()
