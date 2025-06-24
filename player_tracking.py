import os
import sys
import cv2
import numpy as np
import torch
import time
from collections import defaultdict

# Add ByteTrack to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ByteTrack'))

# Import ByteTrack after adding to path
try:
    from bytetrack.byte_tracker import BYTETracker
    from bytetrack.utils.visualize import plot_tracking
except ImportError:
    print("ByteTrack not found. Please install ByteTrack first.")
    print("Run: git clone https://github.com/ifzhang/ByteTrack.git")
    print("cd ByteTrack && pip install -r requirements.txt")
    print("cd ByteTrack && python setup.py develop")
    print("pip install cython_bbox")
    sys.exit(1)

class PlayerTracker:
    def __init__(self, model_path, video_path, output_path=None, conf_thresh=0.5, iou_thresh=0.45):
        """
        Initialize the player tracker
        
        Args:
            model_path: Path to the YOLOv11 (provided model: best.pt) model
            video_path: Path to the input video
            output_path: Path to save the output video (optional)
            conf_thresh: Confidence threshold for detection
            iou_thresh: IOU threshold for NMS
        """
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load YOLOv11 model
        self.model = self._load_model()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer if output path is provided
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        # Initialize ByteTracker
        self.tracker = BYTETracker({
            'track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'frame_rate': self.fps
        })
        
        # Player history for re-identification
        self.player_history = defaultdict(list)
        self.max_history_length = 30  # Store last 30 frames of player data
        
        # Colors for visualization
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    def _load_model(self):
        """
        Load YOLOv11 (provided model: best.pt) model
        """
        try:
            # Try to load using torch.hub first
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path)
        except Exception as e:
            print(f"Error loading model via torch.hub: {e}")
            # Fallback to direct loading
            model = torch.load(self.model_path, map_location='cpu')
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
        
        model.conf = self.conf_thresh
        model.iou = self.iou_thresh
        model.classes = [0]  # Only detect persons
        return model
    
    def _preprocess_detections(self, detections):
        """
        Convert YOLOv11 (provided model: best.pt) detections to ByteTrack format
        """
        if len(detections.xyxy[0]) == 0:
            return np.empty((0, 5))
        
        dets = detections.xyxy[0].cpu().numpy()
        # Format: [x1, y1, x2, y2, confidence]
        return np.array([
            [x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, cls in dets if int(cls) == 0
        ])
    
    def _update_player_history(self, tracks):
        """
        Update player history with current tracks
        """
        for track in tracks:
            track_id = track.track_id
            bbox = track.tlbr
            score = track.score
            
            # Store player data
            self.player_history[track_id].append({
                'bbox': bbox,
                'score': score,
                'features': self._extract_features(bbox)
            })
            
            # Limit history length
            if len(self.player_history[track_id]) > self.max_history_length:
                self.player_history[track_id].pop(0)
    
    def _extract_features(self, bbox):
        """
        Extract simple features from bounding box
        In a real implementation, this would extract appearance features
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        area = width * height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        return np.array([width, height, aspect_ratio, area, center_x, center_y])
    
    def process_video(self):
        """
        Process the video frame by frame
        """
        frame_id = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video with {total_frames} frames...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_id += 1
            if frame_id % 10 == 0:
                print(f"Processing frame {frame_id}/{total_frames}")
            
            # Run YOLOv11 detection
            start_time = time.time()
            results = self.model(frame)
            
            # Convert detections to ByteTrack format
            detections = self._preprocess_detections(results)
            
            # Update tracker
            tracks = self.tracker.update(
                output_results=detections,
                img_info=frame.shape,
                img_size=frame.shape
            )
            
            # Update player history
            self._update_player_history(tracks)
            
            # Visualize results
            online_im = plot_tracking(
                img=frame,
                tlwhs=np.array([track.tlwh for track in tracks]),
                obj_ids=np.array([track.track_id for track in tracks]),
                scores=np.array([track.score for track in tracks]),
                frame_id=frame_id,
                fps=1.0 / (time.time() - start_time)
            )
            
            # Display or save the frame
            if self.writer:
                self.writer.write(online_im)
            
            cv2.imshow('Player Tracking', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        
        print("Video processing completed!")


def main():
    # Define paths
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    video_path = os.path.join(os.path.dirname(__file__), '15sec_input_720p.mp4')
    output_path = os.path.join(os.path.dirname(__file__), 'output_tracking.mp4')
    
    # Create and run player tracker
    tracker = PlayerTracker(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        conf_thresh=0.5,
        iou_thresh=0.45
    )
    
    tracker.process_video()


if __name__ == "__main__":
    main()