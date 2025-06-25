# IdentySync: Player Re-Identification in Sports Footage

This project implements a solution for player re-identification in sports footage. The system ensures that players maintain consistent IDs throughout the video, even when they go out of frame and reappear later.

> **Note**: Large files (model and video files) are not included in this repository due to GitHub size limitations. Please download them separately from the releases section.

## Overview

The player re-identification system uses a combination of object detection (Detection is performed using a fine-tuned Ultralytics YOLOv11 model (`best.pt`)) and tracking with appearance-based re-identification to maintain consistent player identities throughout the video. The system is designed to work in real-time and can handle challenging scenarios such as occlusions, camera movements, and players going in and out of the frame.

## Output

Each player is tracked with a consistent ID even after temporary disappearance. ID labels are visibly overlaid on bounding boxes in the output video.

### Implementation Details

1. **Detection**: Each frame is processed using the fine-tuned Ultralytics YOLOv11 model (`best.pt`) to detect players with high accuracy
2. **Tracking**: A custom tracking algorithm maintains player identities across frames, handling occlusions and fast movements
3. **Re-Identification**: When players disappear and reappear, the system uses deep appearance features to re-identify them and maintain consistent IDs
4. **Visualization**: Each player is assigned a unique color and prominently displayed ID label that remains consistent throughout the video, making it easy to follow individual players

## Features

- Player detection using a fine-tuned Ultralytics YOLOv11 model (`best.pt`)
- Multi-object tracking with occlusion handling
- Appearance-based re-identification using deep features extracted from ResNet18
- Consistent player IDs throughout the video, even when players leave and re-enter the frame
- Enhanced visualization with prominent player ID labels and consistent color-coded bounding boxes
- Real-time processing capabilities
- Comprehensive tracking statistics display (FPS, frame count, gallery size)

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- NumPy
- SciPy
- Git (for cloning ByteTrack)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/PraTham-Patill/IdentySync.git
cd IdentySync
```

2. Download the required large files from the [Releases](https://github.com/PraTham-Patill/IdentySync/releases) section:
   - `best.pt` (YOLOv11 model file)
   - `15sec_input_720p.mp4` (Input video file)
   - Place these files in the root directory of the project

3. Run the setup script to install all dependencies:

```bash
python setup.py
```

This script will:
- Install required Python packages
- Clone and set up ByteTrack
- Check if the model and video files exist

## Usage

To run the player re-identification system:

```bash
python player_reid.py
```

This will process the input video (`15sec_input_720p.mp4`) and generate an output video (`output_reid.mp4`) with tracked players.

For basic tracking without re-identification features:

```bash
python player_tracking.py
```

## Project Structure

- `player_reid.py`: Main implementation with advanced re-identification features
- `player_tracking.py`: Basic implementation using ByteTrack
- `setup.py`: Script to set up dependencies
- `README.md`: Project documentation
- `report.md`: Technical report on the approach and results

## Approach

The system uses a multi-stage approach:

1. **Detection**: A fine-tuned Ultralytics YOLOv11 model (`best.pt`) is used to detect players in each frame with high accuracy
2. **Tracking**: A custom tracking algorithm associates detections across frames to create tracks, handling occlusions and fast movements
3. **Feature Extraction**: Deep features are extracted from player bounding boxes using a pre-trained ResNet18 model
4. **Gallery Management**: Features of tracked players are stored in a gallery for later re-identification
5. **Re-Identification**: When players reappear after leaving the frame, they are compared with the gallery to maintain consistent IDs
6. **Enhanced Visualization**: Players are displayed with prominent ID labels and consistent color-coded bounding boxes for easy tracking

## Customization

You can customize the system by modifying the following parameters in `player_reid.py`:

- `conf_thresh`: Confidence threshold for detection (default: 0.5)
- `iou_thresh`: IOU threshold for NMS (default: 0.45)
- `reid_threshold`: Threshold for re-identification (default: 0.6)
- `max_disappeared_frames`: Maximum frames to keep disappeared players (default: 60)

## License

This project is provided for educational purposes only.

## Acknowledgements

- YOLOv11 (provided model: best.pt) for object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking