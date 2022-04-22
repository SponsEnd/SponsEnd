"""
Main Entry point of the Application
"""

# Importing Libraries and Files.
from flask import Flask, current_app
from src.ml.nlp import find_sponsored_segments
from youtube_transcript_api import YouTubeTranscriptApi
from flask import request
from flask_cors import CORS, cross_origin
from flask import render_template

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello_world():
    return "send_file"


@app.route("/sponsor")
@cross_origin()
def sponsor():
    transcript = getTranscript(request.args.get("id"))
    sponsored_segments = find_sponsored_segments.find_sponsored_segments(transcript)
    return str(sponsored_segments)

def getTranscript(id):
    return YouTubeTranscriptApi.get_transcript(id)