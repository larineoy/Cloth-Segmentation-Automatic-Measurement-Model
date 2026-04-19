from flask import Flask, Response, jsonify, send_from_directory
import subprocess
import time
import os
import threading

app = Flask(__name__)
CAPTURE_DIR = '/home/ubuntu/captures'
os.makedirs(CAPTURE_DIR, exist_ok=True)
# print("Saving to:", path)
# print("Exists after save:", os.path.exists(path))

def generate_frames():
    while True:
        timestamp = str(time.time())
        tmpfile = f'/tmp/stream_{timestamp}.jpg'
        subprocess.run(
            ['fswebcam', '-r', '640x480', '--no-banner', '-q', tmpfile],
            capture_output=True
        )
        if os.path.exists(tmpfile):
            with open(tmpfile, 'rb') as f:
                frame = f.read()
            os.remove(tmpfile)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    filename = f'capture_{int(time.time())}.jpg'
    path = os.path.join(CAPTURE_DIR, filename)
    result = subprocess.run(
        ['fswebcam', '-r', '1920x1080', '--no-banner', path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        return jsonify({'status': 'ok', 'filename': filename})
    return jsonify({'status': 'error', 'msg': result.stderr}), 500

@app.route('/photo/<filename>')
def photo(filename):
    return send_from_directory(CAPTURE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
