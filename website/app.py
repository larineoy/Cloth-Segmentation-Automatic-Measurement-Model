from flask import Flask, render_template, Response, jsonify
import requests

app = Flask(__name__)

PI_URL = "http://100.69.138.166:5001"  # ← replace with your Pi's local IP

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/stream")
def stream():
    def generate():
        with requests.get(f"{PI_URL}/stream", stream=True) as r:
            for chunk in r.iter_content(chunk_size=4096):
                yield chunk
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture", methods=["POST"])
def capture():
    resp = requests.post(f"{PI_URL}/capture")
    data = resp.json()
    if data['status'] == 'ok':
        filename = data['filename']
        # Proxy the image back to the laptop
        img_resp = requests.get(f"{PI_URL}/photo/{filename}")
        with open(f"static/last_capture.jpg", "wb") as f:
            f.write(img_resp.content)
        return jsonify({'status': 'ok', 'photo_url': '/static/last_capture.jpg'})
    return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True)