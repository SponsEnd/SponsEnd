"""
Main Entry point of the Application
"""

# Importing Libraries and Files.
from flask import Flask
from src.ml.nlp import find_sponsored_segments

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to SponsEnd.</p>"


@app.route("/sponsor")
def sponsor():
    find_sponsored_segments.find_sponsored_segments("a. b")
    return "<p>Welcome to SponsEnd.</p>"