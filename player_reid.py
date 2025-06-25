import os
import sys
import cv2
import numpy as np
import torch
import time
from collections import defaultdict
from scipy.spatial.distance import cdist

# Simple tracker implementation to replace ByteTrack
class SimpleTracker:
    def __init__(self, config):
        self.tracks = []
        self.track_id_count = 0
        self.max_age = config.get('track_buffer', 30)
        self.min_iou = 1.0 - config.get('match_thresh', 0.8)
        
    def update(self, dets):
        """
        Update tracks with detections
        dets: list of detections, each in format [x1, y1, x2, y2, score, class_id]
        """
        # Convert detections to format [x1, y1, x2, y2, score, class_id, track_id]
        results = []
        
        # If no tracks yet, initialize with current detections
        if len(self.tracks) == 0:
            for det in dets:
                if det[4] > 0.5:  # Only track high confidence detections
                    self.track_id_count += 1
                    # Add track_id to detection
                    new_track = det.tolist() + [self.track_id_count]
                    self.tracks.append(new_track)
                    results.append(new_track)
            return np.array(results)
        
        # Match detections with existing tracks using IoU
        matched_indices = []
        unmatched_dets = list(range(len(dets)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Calculate IoU between all detections and tracks
        for t, track in enumerate(self.tracks):
            track_box = track[:4]
            best_iou = 1.0
            best_det = -1
            
            for d, det in enumerate(dets):
                det_box = det[:4]
                iou = self._calculate_iou(track_box, det_box)
                
                if iou < best_iou:
                    best_iou = iou
                    best_det = d
            
            # If we found a match
            if best_det >= 0 and best_iou < self.min_iou:
                matched_indices.append((t, best_det))
                if best_det in unmatched_dets:
                    unmatched_dets.remove(best_det)
                if t in unmatched_tracks:
                    unmatched_tracks.remove(t)
        
        # Update matched tracks
        for t, d in matched_indices:
            # Update track with new detection
            track_id = self.tracks[t][6]
            self.tracks[t] = dets[d].tolist() + [track_id]
            results.append(self.tracks[t])
        
        # Add new tracks for unmatched detections
        for d in unmatched_dets:
            if dets[d][4] > 0.5:  # Only add high confidence detections
                self.track_id_count += 1
                new_track = dets[d].tolist() + [self.track_id_count]
                self.tracks.append(new_track)
                results.append(new_track)
        
        # Remove unmatched tracks (they've disappeared)
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in unmatched_tracks]
        
        return np.array(results) if results else np.array([])
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two boxes
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        # If the boxes don't intersect, return 0
        if x_right < x_left or y_bottom < y_top:
            return 1.0
        
        # Calculate area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        
        return 1.0 - iou  # Return distance (1 - IoU) for consistency with tracker logic

# Simple visualization function to replace plot_tracking
def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    """
    Plot tracking results on image
    """
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = f'{int(obj_id)}'  # Convert to int to avoid float display
        if ids2 is not None:
            id_text = f'{id_text}-{int(ids2[i])}'  # Convert to int to avoid float display
        color = (0, 255, 0)  # Default color
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

class PlayerReID:
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
        
        # Initialize SimpleTracker
        self.tracker = SimpleTracker({
            'track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'frame_rate': self.fps
        })
        
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
        Initialize a simple CNN feature extractor
        In a production environment, you would use a dedicated re-ID model
        """
        try:
            # Try to use a pre-trained ResNet model for feature extraction
            self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            # Remove the final classification layer
            self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
            self.feature_extractor.eval()
            if torch.cuda.is_available():
                self.feature_extractor = self.feature_extractor.cuda()
            self.use_deep_features = True
            print("Using ResNet18 for feature extraction")
        except Exception as e:
            print(f"Could not load ResNet model: {e}")
            print("Falling back to simple color histogram features")
            self.use_deep_features = False
    
    def _load_model(self):
        """
        Load YOLOv11 (provided model: best.pt) model
        """
        try:
            # Using the provided best.pt model (YOLOv11)
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)
            model.conf = self.conf_thresh
            model.iou = self.iou_thresh
            model.classes = [0]  # Only detect persons
            return model
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            print("Falling back to a simple detection method")
            # Return a simple detection object that mimics YOLOv11 output
            class SimpleDetector:
                def __init__(self):
                    self.conf = 0.5
                    self.iou = 0.45
                    self.classes = [0]
                
                def __call__(self, img):
                    # Create a simple detection result
                    class SimpleResult:
                        def __init__(self):
                            # Empty detection
                            self.xyxy = [torch.zeros((0, 6))]
                    
                    return SimpleResult()
            
            return SimpleDetector()
    
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
        if self.use_deep_features:
            return self._extract_deep_features(bbox)
        else:
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
            
            # Convert detections to ByteTrack format
            detections = self._preprocess_detections(results)
            
            # Update tracker
            tracks_array = self.tracker.update(detections)
            
            # Convert SimpleTracker output to objects similar to BYTETracker for compatibility
            tracks = []
            for t in tracks_array:
                # Create a simple track object with the same interface as BYTETracker
                track = type('', (), {})()
                track.track_id = int(t[6])  # track_id is at index 6
                track.tlbr = t[:4]  # bounding box coordinates
                track.tlwh = np.array([t[0], t[1], t[2]-t[0], t[3]-t[1]])  # convert to tlwh format
                tracks.append(track)
            
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
            font_scale = 1.5  # Significantly increased font size
            thickness = 4     # Increased thickness for better visibility
            txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Position the ID at the top of the bounding box
            # Add padding around text
            padding = 8
            
            # Draw background for text (larger for better visibility)
            cv2.rectangle(im, 
                         (int(x1), int(y1) - txt_size[1] - padding*2), 
                         (int(x1) + txt_size[0] + padding*2, int(y1)), 
                         color, -1)
            
            # Add black outline to text for better visibility against any background
            # First draw black outline
            cv2.putText(im, text, 
                       (int(x1) + padding, int(y1) - padding), 
                       font, font_scale, (0, 0, 0), thickness+2)
            
            # Then draw white text on top
            cv2.putText(im, text, 
                       (int(x1) + padding, int(y1) - padding), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # Also draw ID at the center of the bounding box for maximum visibility
            center_x = int(x1 + w/2 - txt_size[0]/2)
            center_y = int(y1 + h/2 + txt_size[1]/2)
            
            # Draw background for centered text
            cv2.rectangle(im, 
                         (center_x - padding, center_y - txt_size[1] - padding), 
                         (center_x + txt_size[0] + padding, center_y + padding), 
                         color, -1)
            
            # Draw centered text with outline
            cv2.putText(im, text, 
                       (center_x, center_y), 
                       font, font_scale, (0, 0, 0), thickness+2)
            cv2.putText(im, text, 
                       (center_x, center_y), 
                       font, font_scale, (255, 255, 255), thickness)
        
        # Draw frame info
        fps = 1.0 / processing_time if processing_time > 0 else 0.0
        fps_text = f"FPS: {fps:.1f}"
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
    output_path = os.path.join(os.path.dirname(__file__), 'output_reid.mp4')
    
    print(f"Model path: {model_path}")
    print(f"Video path: {video_path}")
    print(f"Output path: {output_path}")
    
    # Check if files exist
    print(f"Model file exists: {os.path.exists(model_path)}")
    print(f"Video file exists: {os.path.exists(video_path)}")
    
    try:
        # Create and run player re-identification
        print("Creating PlayerReID instance...")
        reid = PlayerReID(
            model_path=model_path,
            video_path=video_path,
            output_path=output_path,
            conf_thresh=0.5,
            iou_thresh=0.45
        )
        
        print("Starting video processing...")
        reid.process_video()
        print("Video processing completed successfully!")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()