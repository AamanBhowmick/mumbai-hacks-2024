import os
import torch
from flask import Flask, request, render_template, send_from_directory, session, redirect, url_for
from moviepy.editor import VideoFileClip, concatenate_videoclips
import cv2
from model import get_caption_model, generate_caption
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PIL import Image
import nltk
import tensorflow as tf
import requests
import uuid
import json
import random
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_bcrypt import Bcrypt
from functools import wraps
from dotenv import load_dotenv

import threading
import queue
import time

load_dotenv()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')


def get_model():
    return get_caption_model()

caption_model = get_model()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def predict(frame):
    print(f"Predicting caption for a video frame...")

    # Convert the OpenCV frame (NumPy array) to a PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Save the frame temporarily as a JPEG (this step might be optional, but keeping it here for compatibility)
    img.save('tmp.jpg')

    captions = []

    # Use the caption model to predict the caption
    pred_caption = generate_caption('tmp.jpg', caption_model)
    captions.append(remove_stopwords(pred_caption))

    # Generate a couple of more captions with noise
    for _ in range(2):
        pred_caption = generate_caption('tmp.jpg', caption_model, add_noise=True)
        if pred_caption not in captions:
            captions.append(remove_stopwords(pred_caption))


    # Remove the temporary image file (optional)
    os.remove('tmp.jpg')

    # Return the generated captions for further use
    return captions


# Initialize Flask App
app = Flask(__name__)

app.secret_key = os.getenv('SECRET_KEY') 

uri =  "mongodb+srv://shree:shree@cluster0.ddync.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
stop_words = set(stopwords.words('english'))
client = MongoClient(uri, server_api=ServerApi('1'))
bcrypt = Bcrypt(app)
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client.test
users = db.users
videos = db.videos

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_trailers'
JSON_UPLOAD_FOLDER = 'output_json'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['JSON_UPLOAD_FOLDER'] = JSON_UPLOAD_FOLDER

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(JSON_UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model for object detection (to help identify action sequences)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

task_queue = queue.Queue()

def background_worker():
    while True:
        # Get the next task from the queue
        task = task_queue.get()
        # if task is None:
        #     break

        id, filepath, json_filepath,genre,frame_skip_rate,duration = task['id'], task['filepath'], task['json_filepath'], task['genre'], task['frame_skip_rate'], task['duration']
        
        # Run the video processing task
        genre_clips = classify_video_by_genres(id,filepath, json_filepath,genre,frame_skip_rate)
        
        for genre, clips in genre_clips.items():
            generate_trailer(id, filepath, clips, genre, duration)
        # Optionally update the status in the database after processing
        
        # Mark the task as done
        task_queue.task_done()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_trailer(id, filepath, clips, demographic, target_duration):
    # Filter out clips with a duration of 0
    print("147: ",clips)
    valid_clips = [(start, end) for start, end in clips if end > start]
    print("149: ",valid_clips)
    if not valid_clips:
        print("No valid clips found.")
        return

    # Randomly select a sequence of consecutive clips that sum up to at least the target duration
    selected_clips = select_clips_to_match_duration(valid_clips, target_duration)
    
    # If no selection matches the duration, merge all the clips as a fallback
    if not selected_clips:
        print(f"Could not find enough clips to match the target duration ({target_duration}s). Merging all clips.")
        selected_clips = valid_clips

    # Concatenate the selected clips to create the trailer
    if selected_clips and len(selected_clips) > 0:
        video_clips = []
        for clip in selected_clips:
            start, end = clip
            video = VideoFileClip(filepath).subclip(start, end)
            video_clips.append(video)
        final_trailer = concatenate_videoclips(video_clips)
        # Save the generated trailer
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], f'{id}_trailer.mp4')
        final_trailer.write_videofile(output_file, codec="libx264")
        video = VideoFileClip(output_file)
        duration = video.duration
        videos.update_one({'id': id}, {'$set': {'status': "Completed",'eta':'00:00','duration':duration}})
        print(f"Generated trailer saved to: {output_file}")
    else:
        print("No clips were selected.")

# Helper function to select consecutive clips that match or exceed the target duration
def select_clips_to_match_duration(clips, target_duration):
    # Calculate the duration of each clip
    clip_durations = [(start, end, end - start) for start, end in clips]

    # Shuffle the clips to introduce randomness in selection
    random.shuffle(clip_durations)

    total_clips = len(clip_durations)
    for i in range(total_clips):
        sum_duration = 0
        selected_clips = []

        # Iterate over the clips starting from index `i` to maintain sequence
        for j in range(i, total_clips):
            start, end, duration = clip_durations[j]
            sum_duration += duration
            selected_clips.append((start, end))

            # If the sum of selected clips' duration meets or exceeds the target, return them
            if sum_duration >= target_duration:
                return selected_clips
    
    # If no sequence matches the duration, return an empty list
    return []


# Function to classify video scenes into genres
def classify_video_by_genres(id,filepath, json_filepath,genre, frame_skip_rate=10000,threshold=5):
    startTime = time.time()
    videos.update_one({'id': id}, {'$set': {'status': "In Progress"}})
    video_capture = cv2.VideoCapture(filepath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    other_clips = []
    
    frame_count = 0
    current_genre = None
    current_clip_start = None
    
    last_clip_end = {
        'Other': -threshold
    }
    frame_count = 0
    
    while frame_count < total_frame_count:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_count % frame_skip_rate == 0:
            results = model(frame)
            detected_objects = results.xyxy[0]
            detected_captions = predict(frame)
            object_list = []
            for obj in detected_objects:
                object_list.append(model.names[int(obj[5])])
            print("Detected captions: ", detected_captions)
            print("Detected objects: ", object_list)

            # Call the API to determine the genre
            current_genre = detect_image_genre(detected_objects, detected_captions,genre)
            print("Detected genre: ", current_genre)

            current_clip_start = frame_count / fps
            current_clip_end = current_clip_start + threshold  # Add the threshold to avoid duplicates

            # Check if the current clip is not a duplicate within the threshold for that genre
            if current_clip_start - last_clip_end.get(current_genre, -threshold) >= threshold:
                # Collect clips for each genre
                if current_genre in genre:
                    other_clips.append((current_clip_start, current_clip_end))

                # Update the last_clip_end for the current genre
                last_clip_end[current_genre] = current_clip_end
                print(f"Added {current_genre} clip from {current_clip_start} to {current_clip_end}")
        print(f"Processing: {frame_count / total_frame_count * 100:.2f}%")
        current_elapsed = time.time() - startTime
        if frame_count == 0:
            eta = 0
        else:
            eta = 100*(current_elapsed / (frame_count / total_frame_count * 100))
        videos.update_one({'id': id}, {'$set': {'eta':eta}})
        frame_count += frame_skip_rate
    video_capture.release()
    data = {
        'other': other_clips
    }
    # Write the data to the JSON file
    with open(json_filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    
    print(f"Clips saved as JSON: {json_filepath}")

    return data

def detect_image_genre(detected_objects, detected_captions,genre):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1",
        "prompt": f"I have a few captions and objects. you have to help me identify in which category the image falls. provide the genre only as response ({'/'.join(genre)}). Detected captions:  {detected_captions} Detected objects:  {detected_objects}. provide the genre only and not any other text. eg: Action. If you are not sure, you can say Other.",
        "stream": False
    }

    # Send the POST request
    response = requests.post(url, json=payload)
    response = response.json()
    return response.get('response')

@app.route("/layout")
def layout():
    user = session.get('user')
    return render_template("layout.html", user = user)

@app.route("/")
def home():
    user = session.get('user')
    return render_template("index.html", user = user)

@app.route("/upload", methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            # Save uploaded video file
            data = request.form
            duration_lst = data.get("duration").split(":")
            duration = int(duration_lst[0]) * 60 + int(duration_lst[1])
            id = str(uuid.uuid4())
            videos.insert_one({'name': data.get('name'), 'id': id, 'genre': data.getlist('genre'),'eta':0,'status':"In Progress",'duration':"", "email": session.get("user")})
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{id}.mp4')
            json_filepath = os.path.join(app.config['JSON_UPLOAD_FOLDER'], f'{id}.json')
            file.save(filepath)

            # Add the processing task to the queue
            task_queue.put({'id': id, 'filepath': filepath, 'json_filepath': json_filepath, 'genre': data.getlist('genre'),'frame_skip_rate': int(data.get('frame-rate')), 'duration': duration * 60})

            return redirect("/history")
    query = request.args.get("view")
    user = session.get('user')
    
    
    return render_template("upload.html", query = query, user=user)


@app.route("/about")
def about():
    user = session.get('user')
    return render_template("about.html", user = user)

@app.route("/contact-us")
def contactus():
    user = session.get('user')
    return render_template("contact-us.html", user=user)

@app.route("/history")
@login_required
def history():
    video_lst = videos.find({"email": session.get("user")})
    user = session.get('user')
    return render_template("history.html", video_lst=list(video_lst), user=user)

# Authentication

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        data = request.form
        login_user = users.find_one({'email' : data.get('email')})
        if login_user:
            password = bcrypt.check_password_hash(login_user['password'], data.get('password'))
            if password:
                session['user'] = str(data.get('email'))
                return redirect("/")
            else:
                return redirect("/")
        return redirect("/")
    return render_template("login.html")

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        hashed = bcrypt.generate_password_hash(data.get('password'))
        curr_user = users.find_one({'email' : data.get('password')})
        if curr_user:
            return redirect("/login")
        new_user = users.insert_one({'email' : data.get('email'), 'password' : hashed})
        if(new_user):
            return redirect("/login")
        else: 
            return redirect("/login")
    return render_template("register.html")

@app.route("/logout", methods=['GET', 'POST'])
def logout():
    session.pop('user', None)  
    return redirect('/login')

# Run the Flask app
worker_thread = threading.Thread(target=background_worker, daemon=True)
worker_thread.start()

if __name__ == "__main__":
    app.run(debug=False)