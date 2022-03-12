"""
Main Entry point of the Application
"""

# Importing Libraries and Files.
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to SponsEnd.</p>"