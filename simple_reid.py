import os
import cv2
import numpy as np
import torch
import time
from collections import defaultdict
from scipy.spatial.distance import cdist

class SimpleTrack:
    """Simple track class to replace ByteTrack"""
    def __init__(self, track_id, bbox, score):
        self.track_id = track_id
        self.tlwh = bbox  # Format: [x, y, w, h]
        self.tlbr = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # Format: [x1, y1, x2, y2]
        self.score = score
        self.time_since_update = 0

class SimpleTracker:
    """Simple tracker to replace ByteTrack"""
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
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                self.tracks.append(SimpleTrack(self.next_id, bbox, score))
                self.next_id += 1
            return self.tracks
        
        # Increment time since update for all tracks
        for track in self.tracks:
            track.time_since_update += 1
        
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
                    iou_matrix[t, d] = self._iou(track.tlbr, det_bbox)
            
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
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                self.tracks[t].tlwh = bbox
                self.tracks[t].tlbr = [x1, y1, x2, y2]
                self.tracks[t].score = score
                self.tracks[t].time_since_update = 0
            
            # Create new tracks for unmatched detections
            for d in unmatched_detections:
                x1, y1, x2, y2, score = detections[d]
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                self.tracks.append(SimpleTrack(self.next_id, bbox, score))
                self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        return self.tracks

class SimplePlayerReID:
    def __init__(self, model_path, video_path, output_path=None, conf_thresh=0.5, iou_thresh=0.45):
        """
        Initialize the player re-identification system
        
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
        
        # Initialize tracker
        self.tracker = SimpleTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Player gallery for re-identification
        self.player_gallery = {}
        self.disappeared_players = {}
        self.reid_threshold = 0.6  # Threshold for re-identification
        self.max_disappeared_frames = 60  # Maximum frames to keep disappeared players
        
        # Colors for visualization
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)
        
        # Frame for feature extraction
        self.current_frame = None
        
        # Initialize feature extractor
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """
        Initialize a simple feature extractor using color histograms
        In a production environment, you would use a dedicated re-ID model
        """
        # Use simple color histogram features for reliability
        print("Using color histogram features for player re-identification")
        self.use_deep_features = False
    
    def _load_model(self):
        """
        Load YOLOv11 (provided model: best.pt) model
        """
        try:
            # First attempt: Try loading with torch.hub with force_reload
            print("Attempting to load YOLOv11 model with torch.hub...")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
            model.conf = self.conf_thresh
            model.iou = self.iou_thresh
            model.classes = [0]  # Only detect persons
            return model
        except Exception as e:
            print(f"Error loading model with torch.hub: {e}")
            print("Falling back to direct model loading...")
            
            # Second attempt: Try loading directly
            try:
                # Load the model directly
                model = torch.load(self.model_path, map_location='cpu')
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                model.conf = self.conf_thresh
                model.iou = self.iou_thresh
                model.classes = [0]  # Only detect persons
                return model
            except Exception as e2:
                print(f"Error loading model directly: {e2}")
                raise RuntimeError(f"Failed to load YOLOv11 model: {e2}")
    
    def _preprocess_detections(self, detections):
        """
        Convert YOLOv11 (provided model: best.pt) detections to tracker format
        """
        if len(detections.xyxy[0]) == 0:
            return np.empty((0, 5))
        
        dets = detections.xyxy[0].cpu().numpy()
        # Format: [x1, y1, x2, y2, confidence]
        return np.array([
            [x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, cls in dets if int(cls) == 0
        ])
    
    def _extract_deep_features(self, bbox):
        """
        Extract deep features from the bounding box using ResNet
        """
        if self.current_frame is None:
            return np.zeros(512)  # Default feature size for ResNet18
        
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bbox is within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512)
        
        # Extract the player patch
        player_patch = self.current_frame[y1:y2, x1:x2]
        if player_patch.size == 0:
            return np.zeros(512)
        
        # Resize and preprocess for the model
        player_patch = cv2.resize(player_patch, (224, 224))
        player_patch = cv2.cvtColor(player_patch, cv2.COLOR_BGR2RGB)
        player_patch = player_patch.transpose(2, 0, 1)  # HWC to CHW
        player_patch = torch.from_numpy(player_patch).float() / 255.0
        player_patch = player_patch.unsqueeze(0)  # Add batch dimension
        
        # Normalize with ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        player_patch = (player_patch - mean) / std
        
        # Extract features
        with torch.no_grad():
            if torch.cuda.is_available():
                player_patch = player_patch.cuda()
            features = self.feature_extractor(player_patch)
            features = features.squeeze().cpu().numpy()
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _extract_color_histogram(self, bbox):
        """
        Extract color histogram features from the bounding box
        """
        if self.current_frame is None:
            return np.zeros(256 * 3)  # 256 bins for each channel
        
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure bbox is within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.width, x2), min(self.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(256 * 3)
        
        # Extract the player patch
        player_patch = self.current_frame[y1:y2, x1:x2]
        if player_patch.size == 0:
            return np.zeros(256 * 3)
        
        # Calculate histogram for each channel
        hist_features = []
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([player_patch], [i], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.append(hist)
        
        # Concatenate histograms
        features = np.concatenate(hist_features)
        return features
    
    def _extract_features(self, bbox):
        """
        Extract features from the bounding box
        """
        # Only use color histogram features for reliability
        return self._extract_color_histogram(bbox)
    
    def _update_player_gallery(self, tracks):
        """
        Update player gallery with current tracks
        """
        # First, increment disappeared count for all players not in current tracks
        current_track_ids = {track.track_id for track in tracks}
        
        # Check for disappeared players
        for track_id in list(self.player_gallery.keys()):
            if track_id not in current_track_ids:
                # Player disappeared
                if track_id not in self.disappeared_players:
                    # First time disappearing, move to disappeared players
                    self.disappeared_players[track_id] = {
                        'features': self.player_gallery[track_id]['features'],
                        'bbox': self.player_gallery[track_id]['bbox'],
                        'disappeared_frames': 0
                    }
                    del self.player_gallery[track_id]
                else:
                    # Already in disappeared players, increment counter
                    self.disappeared_players[track_id]['disappeared_frames'] += 1
        
        # Remove players that have been disappeared for too long
        for track_id in list(self.disappeared_players.keys()):
            if self.disappeared_players[track_id]['disappeared_frames'] > self.max_disappeared_frames:
                del self.disappeared_players[track_id]
        
        # Update gallery with current tracks
        for track in tracks:
            track_id = track.track_id
            bbox = track.tlbr
            
            # Extract features
            features = self._extract_features(bbox)
            
            # Update gallery
            self.player_gallery[track_id] = {
                'features': features,
                'bbox': bbox
            }
    
    def _perform_reid(self, tracks):
        """
        Perform re-identification for tracks that might be disappeared players
        """
        if not self.disappeared_players:
            return tracks
        
        # Get features for all current tracks
        current_features = []
        for track in tracks:
            features = self._extract_features(track.tlbr)
            current_features.append(features)
        
        if not current_features:
            return tracks
        
        # Get features for all disappeared players
        disappeared_ids = list(self.disappeared_players.keys())
        disappeared_features = [self.disappeared_players[id]['features'] for id in disappeared_ids]
        
        # Calculate distance matrix
        distance_matrix = cdist(np.array(current_features), np.array(disappeared_features), 'cosine')
        
        # Find matches
        for i, track in enumerate(tracks):
            # Find the closest disappeared player
            min_idx = np.argmin(distance_matrix[i])
            min_dist = distance_matrix[i][min_idx]
            
            # If distance is below threshold, consider it a match
            if min_dist < self.reid_threshold:
                old_id = track.track_id
                new_id = disappeared_ids[min_idx]
                
                # Update track ID
                track.track_id = new_id
                
                # Remove from disappeared players
                del self.disappeared_players[new_id]
                
                print(f"Re-identified player: {old_id} -> {new_id} (distance: {min_dist:.3f})")
                
                # Update distance matrix to avoid multiple matches to the same disappeared player
                distance_matrix[:, min_idx] = float('inf')
        
        return tracks
    
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
            
            # Store current frame for feature extraction
            self.current_frame = frame.copy()
            
            # Run YOLOv11 detection
            start_time = time.time()
            results = self.model(frame)
            
            # Convert detections to tracker format
            detections = self._preprocess_detections(results)
            
            # Update tracker
            tracks = self.tracker.update(detections)
            
            # Perform re-identification
            tracks = self._perform_reid(tracks)
            
            # Update player gallery
            self._update_player_gallery(tracks)
            
            # Visualize results
            online_im = self._visualize_tracking(frame, tracks, frame_id, time.time() - start_time)
            
            # Display or save the frame
            if self.writer:
                self.writer.write(online_im)
            
            cv2.imshow('Player Re-Identification', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        
        print("Video processing completed!")
    
    def _visualize_tracking(self, image, tracks, frame_id, processing_time):
        """
        Custom visualization of tracking results with enhanced player ID labels
        """
        im = image.copy()
        
        # Draw tracking results
        for track in tracks:
            track_id = track.track_id
            tlwh = track.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Get color for this ID (consistent across frames)
            color = self.colors[track_id % len(self.colors)].tolist()
            
            # Draw bounding box with thicker lines for better visibility
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Draw ID with larger, more visible text
            text = f"ID: {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8  # Increased font size
            thickness = 2
            txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Draw background for text (slightly larger for better visibility)
            cv2.rectangle(im, (int(x1), int(y1) - txt_size[1] - 4), 
                         (int(x1) + txt_size[0] + 4, int(y1)), color, -1)
            
            # Draw text with white color for better contrast
            cv2.putText(im, text, (int(x1) + 2, int(y1) - 4), font, font_scale, (255, 255, 255), thickness)
        
        # Draw frame info
        fps_text = f"FPS: {1.0 / processing_time:.1f}"
        frame_text = f"Frame: {frame_id}"
        cv2.putText(im, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw gallery and disappeared players info
        gallery_text = f"Gallery: {len(self.player_gallery)}"
        disappeared_text = f"Disappeared: {len(self.disappeared_players)}"
        cv2.putText(im, gallery_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, disappeared_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return im


def main():
    # Define paths
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    video_path = os.path.join(os.path.dirname(__file__), '15sec_input_720p.mp4')
    output_path = os.path.join(os.path.dirname(__file__), 'output_simple_reid.mp4')
    
    # Create and run player re-identification
    reid = SimplePlayerReID(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        conf_thresh=0.5,
        iou_thresh=0.45
    )
    
    reid.process_video()


if __name__ == "__main__":
    main()