from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from vehicle_damage import (
    ImageValidationError,
    ModelUnavailableError,
    analyze_image,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/analyze")
def analyze():
    uploaded_file = request.files.get("image")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Analiz etmek için bir görsel seçin."), 400

    try:
        result = analyze_image(uploaded_file.read())
    except ImageValidationError as error:
        return jsonify(error=str(error)), 400
    except ModelUnavailableError as error:
        app.logger.exception("Araç hasar modeli kullanılamıyor.")
        return jsonify(error=str(error)), 503

    return jsonify(result)


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(_error):
    return jsonify(error="Görsel 10 MB sınırını aşıyor."), _error.code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
