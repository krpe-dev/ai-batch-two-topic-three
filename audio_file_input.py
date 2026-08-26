import os
import json
from flask import Flask, request, render_template_string, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from eos_detector import detect_end_of_speech


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Audio File Input</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            padding: 40px;
        }

        .container {
            max-width: 950px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }

        h1 {
            color: #222;
        }

        .upload-box {
            margin-top: 20px;
            padding: 20px;
            border: 2px dashed #999;
            border-radius: 10px;
            background: #fafafa;
        }

        input[type="file"] {
            margin-top: 10px;
            margin-bottom: 15px;
        }

        button {
            background: #2563eb;
            color: white;
            padding: 10px 18px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }

        button:hover {
            background: #1d4ed8;
        }

        button:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }

        .result {
            margin-top: 30px;
            background: #f9fafb;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #2563eb;
        }

        .error {
            margin-top: 30px;
            background: #fee2e2;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #dc2626;
        }

        pre {
            background: #111827;
            color: #e5e7eb;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
        }

        .label {
            font-weight: bold;
            color: #374151;
        }

        .value {
            color: #111827;
        }

        audio {
            width: 100%;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        .small-note {
            color: #6b7280;
            font-size: 14px;
        }

        #loading {
            display: none;
            margin-top: 15px;
            color: #2563eb;
            font-weight: bold;
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid #93c5fd;
            border-top: 2px solid #2563eb;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>Audio File Input</h1>

        <p>
            Upload an audio file. When you click <b>Analyze Audio</b>, the audio will start playing
            while the backend analyzes it using <b>Whisper + Local LLM</b>.
        </p>

        <div class="upload-box">
            <form id="uploadForm">
                <label class="label">Choose audio file:</label><br>

                <input
                    type="file"
                    name="audio"
                    id="audioInput"
                    accept=".wav,.mp3,.m4a"
                    required
                    onchange="previewAudio(event)"
                ><br>

                <p class="small-note">Audio preview:</p>

                <audio id="audioPreview" controls style="display:none;"></audio>

                <button type="submit" id="analyzeBtn">Analyze Audio</button>

                <div id="loading">
                    <span class="spinner"></span>
                    Audio is playing and analysis is running... Please wait.
                </div>
            </form>
        </div>

        <div id="output"></div>
    </div>

    <script>
        const uploadForm = document.getElementById("uploadForm");
        const audioInput = document.getElementById("audioInput");
        const audioPreview = document.getElementById("audioPreview");
        const loading = document.getElementById("loading");
        const output = document.getElementById("output");
        const analyzeBtn = document.getElementById("analyzeBtn");

        function previewAudio(event) {
            const file = event.target.files[0];

            if (file) {
                const audioURL = URL.createObjectURL(file);
                audioPreview.src = audioURL;
                audioPreview.style.display = "block";
            } else {
                audioPreview.style.display = "none";
            }
        }

        function escapeHtml(text) {
            if (text === null || text === undefined) {
                return "";
            }

            return String(text)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }

        uploadForm.addEventListener("submit", async function(event) {
            event.preventDefault();

            const file = audioInput.files[0];

            if (!file) {
                output.innerHTML = `
                    <div class="error">
                        <h2>Error</h2>
                        <p>Please choose an audio file.</p>
                    </div>
                `;
                return;
            }

            output.innerHTML = "";
            loading.style.display = "block";
            analyzeBtn.disabled = true;

            try {
                // Play audio while analysis is running
                audioPreview.currentTime = 0;
                await audioPreview.play();

                const formData = new FormData();
                formData.append("audio", file);

                const response = await fetch("/analyze", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                loading.style.display = "none";
                analyzeBtn.disabled = false;

                if (!data.success) {
                    output.innerHTML = `
                        <div class="error">
                            <h2>Error</h2>
                            <p>${escapeHtml(data.error)}</p>
                        </div>
                    `;
                    return;
                }

                const result = data.result;
                const resultJson = JSON.stringify(result, null, 2);

                output.innerHTML = `
                    <div class="result">
                        <h2>Detection Result</h2>

                        <p>
                            <span class="label">Status:</span>
                            <span class="value">${escapeHtml(result.status)}</span>
                        </p>

                        <p>
                            <span class="label">End of Speech:</span>
                            <span class="value">${escapeHtml(result.end_of_speech)}</span>
                        </p>

                        <p>
                            <span class="label">Confidence:</span>
                            <span class="value">${escapeHtml(result.confidence)}</span>
                        </p>

                        <p>
                            <span class="label">Reason:</span>
                            <span class="value">${escapeHtml(result.reason)}</span>
                        </p>

                        <p>
                            <span class="label">Transcript:</span>
                        </p>

                        <pre>${escapeHtml(result.transcript)}</pre>

                        <h3>Full JSON Output</h3>
                        <pre>${escapeHtml(resultJson)}</pre>
                    </div>
                `;

            } catch (error) {
                loading.style.display = "none";
                analyzeBtn.disabled = false;

                output.innerHTML = `
                    <div class="error">
                        <h2>Error</h2>
                        <p>${escapeHtml(error.message)}</p>
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template_string(HTML_PAGE)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file uploaded."
        })

    file = request.files["audio"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No selected file."
        })

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Invalid file type. Please upload WAV, MP3, or M4A."
        })

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    file.save(file_path)

    try:
        result = detect_end_of_speech(file_path)

        return jsonify({
            "success": True,
            "result": result,
            "audio_url": f"/uploads/{filename}"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)