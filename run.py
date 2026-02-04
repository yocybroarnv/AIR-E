"""
Simple launcher script for AIR-E
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    
    # Run streamlit app
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
