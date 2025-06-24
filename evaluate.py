import os
import sys
import cv2
import numpy as np
import time
import argparse
from collections import defaultdict

# Import player re-identification system
from player_reid import PlayerReID

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Player Re-Identification')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to YOLOv5 model')
    parser.add_argument('--video', type=str, default='15sec_input_720p.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_evaluation.mp4', help='Path to output video')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--reid-thresh', type=float, default=0.6, help='Threshold for re-identification')
    return parser.parse_args()

class Evaluator:
    def __init__(self, reid_system):
        """
        Initialize the evaluator
        
        Args:
            reid_system: Initialized PlayerReID instance
        """
        self.reid_system = reid_system
        self.track_history = defaultdict(list)  # Track history for each ID
        self.id_switches = 0  # Count of ID switches
        self.reid_events = 0  # Count of re-identification events
        self.total_tracks = 0  # Total number of tracks
        self.frame_processing_times = []  # List of frame processing times
    
    def evaluate(self):
        """
        Evaluate the player re-identification system
        """
        frame_id = 0
        total_frames = int(self.reid_system.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Evaluating on {total_frames} frames...")
        
        while True:
            ret, frame = self.reid_system.cap.read()
            if not ret:
                break
            
            frame_id += 1
            if frame_id % 10 == 0:
                print(f"Processing frame {frame_id}/{total_frames}")
            
            # Store current frame for feature extraction
            self.reid_system.current_frame = frame.copy()
            
            # Run YOLOv5 detection
            start_time = time.time()
            results = self.reid_system.model(frame)
            
            # Convert detections to ByteTrack format
            detections = self.reid_system._preprocess_detections(results)
            
            # Update tracker
            tracks = self.reid_system.tracker.update(
                output_results=detections,
                img_info=frame.shape,
                img_size=frame.shape
            )
            
            # Record track IDs before re-identification
            track_ids_before = {track.track_id for track in tracks}
            
            # Perform re-identification
            tracks = self.reid_system._perform_reid(tracks)
            
            # Record track IDs after re-identification
            track_ids_after = {track.track_id for track in tracks}
            
            # Count re-identification events
            self.reid_events += len(track_ids_before - track_ids_after)
            
            # Update player gallery
            self.reid_system._update_player_gallery(tracks)
            
            # Update track history and count ID switches
            self._update_track_history(tracks)
            
            # Record frame processing time
            processing_time = time.time() - start_time
            self.frame_processing_times.append(processing_time)
            
            # Visualize results
            online_im = self._visualize_evaluation(frame, tracks, frame_id, processing_time)
            
            # Display or save the frame
            if self.reid_system.writer:
                self.reid_system.writer.write(online_im)
            
            cv2.imshow('Player Re-Identification Evaluation', online_im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.reid_system.cap.release()
        if self.reid_system.writer:
            self.reid_system.writer.release()
        cv2.destroyAllWindows()
        
        # Print evaluation results
        self._print_evaluation_results()
    
    def _update_track_history(self, tracks):
        """
        Update track history and count ID switches
        """
        for track in tracks:
            track_id = track.track_id
            bbox = track.tlbr
            
            # Calculate center point of bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Check if this is a new track
            if len(self.track_history[track_id]) == 0:
                self.total_tracks += 1
            
            # Check for potential ID switch (sudden position change)
            if len(self.track_history[track_id]) > 0:
                prev_x, prev_y = self.track_history[track_id][-1]
                dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                
                # If distance is too large, it might be an ID switch
                if dist > 100:  # Threshold for position change
                    self.id_switches += 1
            
            # Update track history
            self.track_history[track_id].append((center_x, center_y))
    
    def _visualize_evaluation(self, image, tracks, frame_id, processing_time):
        """
        Visualize evaluation results
        """
        im = image.copy()
        
        # Draw tracking results
        for track in tracks:
            track_id = track.track_id
            tlwh = track.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            # Get color for this ID
            color = self.reid_system.colors[track_id % len(self.reid_system.colors)].tolist()
            
            # Draw bounding box
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw ID
            text = f"ID: {track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
            cv2.rectangle(im, (int(x1), int(y1) - txt_size[1] - 2), 
                         (int(x1) + txt_size[0], int(y1)), color, -1)
            cv2.putText(im, text, (int(x1), int(y1) - 2), font, 0.5, (255, 255, 255), 2)
            
            # Draw track history
            if len(self.track_history[track_id]) > 1:
                for i in range(1, len(self.track_history[track_id])):
                    pt1 = (int(self.track_history[track_id][i-1][0]), int(self.track_history[track_id][i-1][1]))
                    pt2 = (int(self.track_history[track_id][i][0]), int(self.track_history[track_id][i][1]))
                    cv2.line(im, pt1, pt2, color, 2)
        
        # Draw evaluation metrics
        fps_text = f"FPS: {1.0 / processing_time:.1f}"
        frame_text = f"Frame: {frame_id}"
        tracks_text = f"Tracks: {self.total_tracks}"
        switches_text = f"ID Switches: {self.id_switches}"
        reid_text = f"ReID Events: {self.reid_events}"
        
        cv2.putText(im, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, frame_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, tracks_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, switches_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(im, reid_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return im
    
    def _print_evaluation_results(self):
        """
        Print evaluation results
        """
        avg_fps = 1.0 / np.mean(self.frame_processing_times) if self.frame_processing_times else 0
        
        print("\nEvaluation Results:")
        print(f"Total Tracks: {self.total_tracks}")
        print(f"ID Switches: {self.id_switches}")
        print(f"Re-Identification Events: {self.reid_events}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Average Processing Time: {np.mean(self.frame_processing_times)*1000:.2f} ms")


def main():
    # Parse arguments
    args = parse_args()
    
    # Define paths
    model_path = os.path.join(os.path.dirname(__file__), args.model)
    video_path = os.path.join(os.path.dirname(__file__), args.video)
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    
    # Create player re-identification system
    reid = PlayerReID(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        conf_thresh=args.conf,
        iou_thresh=args.iou
    )
    
    # Set re-identification threshold
    reid.reid_threshold = args.reid_thresh
    
    # Create evaluator
    evaluator = Evaluator(reid)
    
    # Run evaluation
    evaluator.evaluate()


if __name__ == "__main__":
    main()