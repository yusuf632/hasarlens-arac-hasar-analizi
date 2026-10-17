from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from vehicle_damage import ImageValidationError, ModelUnavailableError, analyze_image


TEST_IMAGE_PATH = Path(__file__).resolve().with_name("test13.jpg")


def run_pro_expert_system() -> None:
    print(">>> AI Analiz Motoru Başlatılıyor...")

    if not TEST_IMAGE_PATH.is_file():
        print(f"!! {TEST_IMAGE_PATH.name} bulunamadı!")
        return

    try:
        result = analyze_image(TEST_IMAGE_PATH.read_bytes())
    except (ImageValidationError, ModelUnavailableError, OSError) as error:
        print(f"!! Analiz başarısız: {error}")
        return

    with Image.open(TEST_IMAGE_PATH) as image:
        original_image = image.convert("RGB").copy()

    label = str(result["label"])
    confidence = float(result["confidence"])
    analyzed_at = str(result["analyzed_at"])
    is_damaged = bool(result["is_damaged"])

    plt.style.use("dark_background")
    _, axes = plt.subplots(figsize=(10, 8))
    axes.imshow(original_image)
    axes.axis("off")

    main_color = "#FF0000" if is_damaged else "#00FF00"
    plt.title(
        "YAPAY ZEKA HASAR ANALİZ SİSTEMİ",
        fontsize=16,
        fontweight="bold",
        color="white",
        pad=30,
    )

    status = "HASAR SAPTANDI" if is_damaged else "HASAR SAPTANMADI"
    report_text = (
        f" DURUM: {status}\n"
        " ━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f" BÖLGE: {label}\n"
        f" GÜVEN: %{confidence:.2f}\n"
        f" ANALİZ ZAMANI: {analyzed_at}"
    )
    axes.text(
        0.02,
        0.98,
        report_text,
        transform=axes.transAxes,
        fontsize=12,
        color="white",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(
            facecolor="black",
            alpha=0.8,
            edgecolor=main_color,
            lw=3,
            boxstyle="round,pad=1",
        ),
    )
    axes.text(
        0.98,
        0.02,
        "AI SCAN APPROVED",
        transform=axes.transAxes,
        fontsize=16,
        color=main_color,
        fontweight="black",
        alpha=0.4,
        ha="right",
        va="bottom",
        rotation=15,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pro_expert_system()
