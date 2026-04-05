import os
import sys
sys.path.append(os.path.dirname(__file__))

# Import the Flask app
from backend.app import app

# Firebase Functions wrapper
from firebase_functions import https_fn

@https_fn.on_request()
def api(req):
    """Firebase Functions wrapper for the Flask app."""
    return https_fn.Response(app)