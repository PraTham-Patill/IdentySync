import os
import cv2
import numpy as np
import torch
import time

class SimpleTracker:
    """Simple tracker for player tracking"""
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        # Convert to [x1, y1, x2, y2] format if needed
        if len(bbox1) == 4 and bbox1[2] < bbox1[0] + bbox1[2]:
            bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
        if len(bbox2) == 4 and bbox2[2] < bbox2[0] + bbox2[2]:
            bbox2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]
        
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # If no tracks yet, initialize with all detections
        if len(self.tracks) == 0 and len(detections) > 0:
            for i, det in enumerate(detections):
                x1, y1, x2, y2, score = det
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'time_since_update': 0
                })
                self.next_id += 1
            return self.tracks
        
        # Increment time since update for all tracks
        for track in self.tracks:
            track['time_since_update'] += 1
        
        # Match detections to existing tracks based on IoU
        if len(detections) > 0:
            matched_indices = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))
            
            # Calculate IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for t, track in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    x1, y1, x2, y2, _ = det
                    det_bbox = [x1, y1, x2, y2]  # [x1, y1, x2, y2]
                    iou_matrix[t, d] = self._iou(track['bbox'], det_bbox)
            
            # Find matches using greedy assignment
            while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
                # Find max IoU
                max_iou = 0
                max_t = -1
                max_d = -1
                
                for t in unmatched_tracks:
                    for d in unmatched_detections:
                        if iou_matrix[t, d] > max_iou:
                            max_iou = iou_matrix[t, d]
                            max_t = t
                            max_d = d
                
                if max_iou >= self.iou_threshold:
                    matched_indices.append((max_t, max_d))
                    unmatched_tracks.remove(max_t)
                    unmatched_detections.remove(max_d)
                else:
                    break
            
            # Update matched tracks
            for t, d in matched_indices:
                x1, y1, x2, y2, score = detections[d]
                self.tracks[t]['bbox'] = [x1, y1, x2, y2]
                self.tracks[t]['score'] = score
                self.tracks[t]['time_since_update'] = 0
            
            # Create new tracks for unmatched detections
            for d in unmatched_detections:
                x1, y1, x2, y2, score = detections[d]
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'time_since_update': 0
                })
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= self.max_age]
        
        return self.tracks

class SimplePlayerTracking:
    def __init__(self, video_path, output_path=None, conf_thresh=0.5, iou_thresh=0.45):
        """Initialize the player tracking system"""
        self.video_path = video_path
        self.output_path = output_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Load YOLOv11 model (provided model: best.pt)
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
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Colors for visualization
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
    
    def _load_model(self):
        """Load YOLOv11 model (provided model: best.pt)"""
        try:
            print("Loading YOLOv11 model (best.pt) from local path...")
            # Use the provided best.pt model
            model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
            try:
                # Try to load using torch.hub first
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
            except Exception as e:
                print(f"Error loading model via torch.hub: {e}")
                # Fallback to direct loading
                model = torch.load(model_path, map_location='cpu')
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
            
            model.conf = self.conf_thresh  # NMS confidence threshold
            model.iou = self.iou_thresh    # NMS IoU threshold
            model.classes = [0]            # Only detect persons (class 0 in COCO)
            model.max_det = 100            # Maximum number of detections per image
            return model
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            raise RuntimeError(f"Failed to load YOLOv11 model: {e}")
    
    def _preprocess_detections(self, detections):
        """Convert YOLOv11 (provided model: best.pt) detections to tracker format"""
        if len(detections.xyxy[0]) == 0:
            return np.empty((0, 5))
        
        dets = detections.xyxy[0].cpu().numpy()
        # Format: [x1, y1, x2, y2, confidence]
        return np.array([
            [x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, cls in dets if int(cls) == 0
        ])
    
    def process_video(self):
        """Process the video frame by frame"""
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
            
            # Run YOLOv5 detection
            start_time = time.time()
            results = self.model(frame)
            
            # Convert detections to tracker format
            detections = self._preprocess_detections(results)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Visualize results
            online_im = self._visualize_tracking(frame, tracks, frame_id, time.time() - start_time)
            
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
    
    def _visualize_tracking(self, image, tracks, frame_id, processing_time):
        """Custom visualization of tracking results"""
        im = image.copy()
        
        # Draw tracking results
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            
            # Get color for this ID (consistent across frames)
            color = self.colors[track_id % len(self.colors)].tolist()
            
            # Draw bounding box
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw ID
            text = f"ID: {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            cv2.rectangle(im, (int(x1), int(y1) - txt_size[1] - 2), 
                         (int(x1) + txt_size[0], int(y1)), color, -1)
            cv2.putText(im, text, (int(x1), int(y1) - 2), font, 0.5, (255, 255, 255), 2)
        
        # Draw frame info
        fps_text = f"FPS: {1.0 / processing_time:.1f}"
        frame_text = f"Frame: {frame_id}"
        cv2.putText(im, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw track count
        tracks_text = f"Tracks: {len(tracks)}"
        cv2.putText(im, tracks_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return im


def main():
    # Define paths
    video_path = os.path.join(os.path.dirname(__file__), '15sec_input_720p.mp4')
    output_path = os.path.join(os.path.dirname(__file__), 'output_simple_tracking.mp4')
    
    # Create and run player tracking
    tracker = SimplePlayerTracking(
        video_path=video_path,
        output_path=output_path,
        conf_thresh=0.5,
        iou_thresh=0.45
    )
    
    tracker.process_video()


if __name__ == "__main__":
    main()