import os
from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging
import face_recognition
from flask_socketio import SocketIO
import base64


logging.getLogger('socketio').setLevel(logging.DEBUG)
logging.getLogger('engineio').setLevel(logging.DEBUG)



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")


# Base de données de visages avec ID associé
faces = {}
UPLOAD_FOLDER = 'photos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


images = []
classNames = []

# Load known faces and encode them
def load_known_faces(UPLOAD_FOLDER):
    personsList = os.listdir(UPLOAD_FOLDER)
    for cl in personsList:
        curImage = cv2.imread(f'{UPLOAD_FOLDER}/{cl}')
        images.append(curImage)
        classNames.append(os.path.splitext(cl)[0])

def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

load_known_faces(UPLOAD_FOLDER)
encode_list_known = find_encodings(images)

def recognize_faces(frame):
    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    faces_data = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        faces_data.append({
            "top": top, "right": right, "bottom": bottom, "left": left, "name": name
        })


@socketio.on('frame')
def handle_frame(data):
    try:
        # Directly access 'frame' assuming data is properly structured.
        # This removes one layer of .get() which seemed incorrect according to the error message.
        encoded_image = data['frame']  # Direct access if you're sure of the structure
        
        # Validate encoded_image format
        if not isinstance(encoded_image, str) or not encoded_image.startswith('data:image/jpeg;base64,'):
            print("Received data is not in the expected format.")
            return
        
        # Process the base64 encoded image
        base64_encoded_data = encoded_image.split('data:image/jpeg;base64,')[1]
        image_data = base64.b64decode(base64_encoded_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Ensure you're using `img` instead of `frame` for resizing and further processing
        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        
        # Your face recognition processing
        face_locations = face_recognition.face_locations(imgS)
        face_encodings = face_recognition.face_encodings(imgS, face_locations)
        faces_data = []

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, face_encoding)
            face_distances = face_recognition.face_distance(encode_list_known, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = classNames[best_match_index].upper()  # Recognized face name
                recognized = True
            else:
                name = "Unknown"
                recognized = False

            # Adjust scaling factor due to image resizing, moved inside the loop
            scale_factor = 4
            top *= scale_factor
            right *= scale_factor
            bottom *= scale_factor
            left *= scale_factor

            faces_data.append({
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                "name": name,
                "recognized": recognized
            })
            print(faces_data)

        # Emit processed face data
        socketio.emit('faces_detected', {'faces': faces_data})  # Ensure data is structured as expected by the client

        print("Image processed successfully.")

    except Exception as e:
        print(f"Error processing frame: {e}")



# Route pour ajouter un visage
@app.route('/faces', methods=['POST'])
def add_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    face_id = request.form['name']

    # Save the image with the provided name
    image_path = os.path.join(UPLOAD_FOLDER, face_id + ".jpg")
    image_file.save(image_path)

    # Convert the image to numpy array
    image = cv2.imread(image_path)

    # Store the image in the 'faces' dictionary
    faces[face_id] = image

    return jsonify({"message": "Face added successfully"}), 201







if __name__ == '__main__':
  socketio.run(app, host='0.0.0.0', port=5001)

