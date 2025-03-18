import cv2
import numpy as np
import math
import argparse
from typing import Dict, List, Tuple

class SimpleFaceProfiler:
    def __init__(self):
        self.face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        self.eye_cascade =cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
        self.smile_cascade= cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_smile.xml')

        self.personality_traits = {
            'face_width_height_ratio': {
                'high':"Assertive and achievement-oriented",
                'medium':"Balanced approach to challenges",
                'low':"Thoughtful and contemplative"
            },
            'eye_face_ratio': {
                'high':"Observant and detail-oriented",
                'medium': "Balanced perception and focus",
                'low':"Big-picture thinker, may overlook details"
            },
            'eye_spacing_ratio': {
                'high': "Open-minded and receptive to new ideas",
                'medium': "Practical and grounded",
                'low': "Focused and determined"
            },
            'smile_width_ratio': {
                'high': "Outgoing and sociable",
                'medium': "Balanced between social and private",
                'low': "Reserved and thoughtful"
            }
        }
    
    def detect_features(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            print("No face detected in the image.")
            return image, {}
        
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract face region
        face_roi = gray[y:y+h, x:x+w]
        face_roi_color = image[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 4)
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Filter out false positives by size and position
            if 0.05 * h < eh < 0.3 * h and 0.05 * w < ew < 0.3 * w:
                valid_eyes.append((ex, ey, ew, eh))
        
        valid_eyes.sort(key=lambda e: e[0])
        
        if len(valid_eyes) > 2:
            # If more than 2 eyes detected, keep the largest two
            valid_eyes.sort(key=lambda e: e[2] * e[3], reverse=True)
            valid_eyes = valid_eyes[:2]
            # Re-sort by x-coordinate
            valid_eyes.sort(key=lambda e: e[0])
        
        eye_centers = []
        for (ex, ey, ew, eh) in valid_eyes:
            cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            eye_center = (x + ex + ew//2, y + ey + eh//2)
            eye_centers.append(eye_center)
            cv2.circle(image, eye_center, 3, (255, 0, 0), -1)
        
        smiles = self.smile_cascade.detectMultiScale(face_roi, 1.8, 20)
        mouth_rect = None
        if len(smiles) > 0:
            # Get the largest smile in the lower half of the face
            valid_smiles = []
            for (sx, sy, sw, sh) in smiles:
                if sy > h/2:  # Only consider detections in the lower half of the face
                    valid_smiles.append((sx, sy, sw, sh))
            
            if valid_smiles:
                # Get the largest smile
                mouth_rect = max(valid_smiles, key=lambda s: s[2] * s[3])
                sx, sy, sw, sh = mouth_rect
                cv2.rectangle(face_roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        
        features = {}
        features['face_width'] = w
        features['face_height'] = h
        features['face_width_height_ratio'] = w / h if h > 0 else 0
        
        if len(eye_centers) == 2:
            left_eye, right_eye = eye_centers
            eye_distance = math.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            features['eye_distance'] = eye_distance
            features['eye_spacing_ratio'] = eye_distance / w if w > 0 else 0
            
            if len(valid_eyes) == 2:
                avg_eye_width = (valid_eyes[0][2] + valid_eyes[1][2]) / 2
                avg_eye_height = (valid_eyes[0][3] + valid_eyes[1][3]) / 2
                features['avg_eye_width'] = avg_eye_width
                features['avg_eye_height'] = avg_eye_height
                features['eye_face_ratio'] = (avg_eye_width * avg_eye_height) / (w * h) if w * h > 0 else 0
        
        if mouth_rect:
            sx, sy, sw, sh = mouth_rect
            features['mouth_width'] = sw
            features['mouth_height'] = sh
            features['smile_width_ratio'] = sw / w if w > 0 else 0
        
        y_text = y + h + 20
        cv2.putText(image, f"Face Width: {w}px", (10, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_text += 25
        cv2.putText(image, f"Face Height: {h}px", (10, y_text), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_text += 25
        if 'eye_distance' in features:
            cv2.putText(image, f"Eye Distance: {features['eye_distance']:.1f}px", (10, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_text += 25
        if 'face_width_height_ratio' in features:
            cv2.putText(image, f"Face Ratio (W/H): {features['face_width_height_ratio']:.2f}", (10, y_text), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return image, features
    def analyze_personality(self, features: Dict) -> Dict:
        if not features:
            return {}
        thresholds = {
            'face_width_height_ratio': {'low': 0.7, 'high': 0.85},
            'eye_face_ratio': {'low': 0.01, 'high': 0.03},
            'eye_spacing_ratio': {'low': 0.3, 'high': 0.45},
            'smile_width_ratio': {'low': 0.4, 'high': 0.6}
        }
        categories = {}
        for trait, threshold in thresholds.items():
            if trait in features:
                if features[trait] < threshold['low']:
                    categories[trait] = 'low'
                elif features[trait] > threshold['high']:
                    categories[trait] = 'high'
                else:
                    categories[trait] = 'medium'
                personality = {}
        for trait, category in categories.items():
            if trait in self.personality_traits and category in self.personality_traits[trait]:
                personality[trait] = self.personality_traits[trait][category]
        
        return personality
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, Dict, Dict]:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, {}, {}
        max_dim = 1024
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
        result_image, features = self.detect_features(image)
        
        if not features:
            return result_image, {}, {}
        personality = self.analyze_personality(features)
        return result_image, features, personality
    
    def display_results(self, image: np.ndarray, features: Dict, personality: Dict) -> None:
        cv2.imshow("Face Profiling Results", image)
        print("\nFACIAL MEASUREMENTS ")
        for key, value in features.items():
            if isinstance(value, (int, float)):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
        print("\nPERSONALITY ANALYSIS")
        
        for trait, description in personality.items():
            print(f"{trait.replace('_', ' ').title()}: {description}")
        print("\nPress any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simple Face Profiling System')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()
    profiler = SimpleFaceProfiler()
    result_image, features, personality = profiler.process_image(args.image)
    if result_image is not None:
        profiler.display_results(result_image, features, personality)
        output_path = 'face_profiling_result.jpg'
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to {output_path}")

if __name__ == "__main__":
    main()