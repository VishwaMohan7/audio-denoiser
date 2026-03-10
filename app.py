import os
import time
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
from backend.inference import AudioDenoiseInference
from backend.visualize import generate_visualizations

app = Flask(__name__, static_folder="frontend")

# Configure upload and output folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__name__)), "data", "uploads")
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__name__)), "data", "outputs")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__name__)), "models", "denoise_model.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Initialize Inference Model
print("Initializing model for inference...")
inferencer = AudioDenoiseInference(MODEL_PATH)
print("Model initialized.")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/denoise", methods=["POST"])
def denoise_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
        
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        # Create unique names to prevent overwriting during concurrent usage
        timestamp = int(time.time())
        input_filename = f"{timestamp}_{filename}"
        output_filename = f"{timestamp}_clean_{filename}"
        
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        file.save(input_path)
        
        try:
            # Process the audio file (Denoise)
            inferencer.denoise_file(input_path, output_path)
            
            # Generate the visualization images
            wf_name, spec_name = generate_visualizations(input_path, output_path, OUTPUT_FOLDER, timestamp)
            
            # Return the processed file path references
            return jsonify({
                "success": True, 
                "message": "Audio denoised successfully",
                "clean_audio_url": f"/api/download/{output_filename}",
                "waveform_url": f"/api/download/{wf_name}",
                "spectrogram_url": f"/api/download/{spec_name}"
            })
        except Exception as e:
            print(f"Error during denoising: {e}")
            return jsonify({"error": str(e)}), 500

@app.route("/api/download/<filename>")
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=False)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
