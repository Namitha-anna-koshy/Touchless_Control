from flask import Flask, render_template, jsonify, send_from_directory
import subprocess
import threading
import os
import signal
import sys
import cv2  # Add OpenCV to check for camera

app = Flask(__name__, static_folder='.', static_url_path='')

# Global variable to track the controller process
controller_process = None

def check_camera_available():
    """Check if a camera is available and working"""
    try:
        # Try to open the default camera (usually 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        # Try to read a frame to confirm camera is working
        ret, frame = cap.read()
        if not ret:
            return False
            
        # Release the camera
        cap.release()
        return True
    except Exception as e:
        print(f"Error checking camera: {e}")
        return False

def run_gesture_controller():
    """Function to run the gesture controller script"""
    global controller_process
    try:
        # Use subprocess to run the gesture controller
        # Make sure to use the correct path to your controller script
        controller_process = subprocess.Popen([sys.executable, 'gesture_controller.py'],
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
        print("Gesture controller started!")
        
        # You can handle the output if needed
        # stdout, stderr = controller_process.communicate()
        # print(f"Output: {stdout.decode()}")
        # print(f"Error: {stderr.decode()}")
    except Exception as e:
        print(f"Error starting gesture controller: {e}")
        return False
    
    return True

@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/start-gesture-controller', methods=['POST'])
def start_controller():
    """API endpoint to start the gesture controller"""
    global controller_process
    
    # First check if a camera is available
    if not check_camera_available():
        return jsonify({
            "status": "error", 
            "message": "No camera connected or camera is in use by another application"
        }), 400
    
    # Kill existing process if there is one
    if controller_process is not None:
        try:
            os.kill(controller_process.pid, signal.SIGTERM)
            controller_process = None
        except:
            pass
    
    # Start the controller in a separate thread so it doesn't block the server
    thread = threading.Thread(target=run_gesture_controller)
    thread.daemon = True
    thread.start()
    
    # Return success response
    return jsonify({"status": "started", "message": "Gesture controller is running"})

@app.route('/stop-gesture-controller', methods=['POST'])
def stop_controller():
    """API endpoint to stop the gesture controller"""
    global controller_process
    
    if controller_process is not None:
        try:
            os.kill(controller_process.pid, signal.SIGTERM)
            controller_process = None
            return jsonify({"status": "stopped", "message": "Gesture controller stopped"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error stopping controller: {e}"})
    else:
        return jsonify({"status": "not_running", "message": "Gesture controller is not running"})

@app.route('/check-camera', methods=['GET'])
def check_camera():
    """API endpoint to check if camera is available"""
    if check_camera_available():
        return jsonify({"status": "available", "message": "Camera is available"})
    else:
        return jsonify({"status": "unavailable", "message": "No camera detected or camera is in use"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)