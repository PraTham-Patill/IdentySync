import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 6):
        print("Error: Python 3.6 or higher is required.")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    requirements = [
        "torch",
        "torchvision",
        "opencv-python",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "Pillow",
        "PyYAML",
        "cython-bbox"
    ]
    
    print("Installing required packages...")
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All required packages installed successfully!")

def setup_bytetrack():
    """Clone and setup ByteTrack"""
    if os.path.exists("ByteTrack"):
        print("ByteTrack directory already exists. Skipping clone.")
    else:
        print("Cloning ByteTrack repository...")
        subprocess.check_call(["git", "clone", "https://github.com/ifzhang/ByteTrack.git"])
    
    print("Installing ByteTrack requirements...")
    os.chdir("ByteTrack")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("Setting up ByteTrack...")
    subprocess.check_call([sys.executable, "setup.py", "develop"])
    
    os.chdir("..")
    print("ByteTrack setup completed!")

def check_model_and_video():
    """Check if model and video files exist"""
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    video_path = os.path.join(os.path.dirname(__file__), '15sec_input_720p.mp4')
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
    else:
        print(f"Model file found at {model_path}")
    
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
    else:
        print(f"Video file found at {video_path}")

def main():
    print("Setting up Player Re-Identification System...")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    check_python_version()
    install_requirements()
    setup_bytetrack()
    check_model_and_video()
    
    print("\nSetup completed successfully!")
    print("You can now run the player re-identification system using:")
    print("python player_reid.py")

if __name__ == "__main__":
    main()