import os
import cv2
import numpy as np

def generate_synthetic_face(gender_type, seed=0):
    """
    Generate a simple face-like synthetic image for testing purposes.
    """
    np.random.seed(seed)
    
    # 224x224 RGB image
    img = np.full((224, 224, 3), (240, 235, 230), dtype=np.uint8)
    
    # Background slight gradient/texture
    bg_noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + bg_noise, 0, 255).astype(np.uint8)
    
    center = (112, 112)
    axes = (65 + np.random.randint(-3, 4), 85 + np.random.randint(-3, 4))
    
    # Base skin color tone (BGR)
    skin_b = 160 + np.random.randint(-20, 20)
    skin_g = 190 + np.random.randint(-20, 20)
    skin_r = 230 + np.random.randint(-15, 15)
    cv2.ellipse(img, center, axes, 0, 0, 360, (skin_b, skin_g, skin_r), -1)
    
    # Eyes
    cv2.circle(img, (85, 95), 10, (255, 255, 255), -1)
    cv2.circle(img, (139, 95), 10, (255, 255, 255), -1)
    cv2.circle(img, (85, 95), 4, (40, 30, 20), -1)
    cv2.circle(img, (139, 95), 4, (40, 30, 20), -1)
    
    # Eyebrows
    eyebrow_thickness = 3 if gender_type == 'male' else 2
    cv2.line(img, (73, 80), (97, 82), (30, 20, 10), eyebrow_thickness)
    cv2.line(img, (127, 82), (151, 80), (30, 20, 10), eyebrow_thickness)
    
    # Nose
    cv2.line(img, (112, 105), (110, 125), (skin_b - 30, skin_g - 30, skin_r - 30), 2)
    cv2.line(img, (110, 125), (117, 125), (skin_b - 30, skin_g - 30, skin_r - 30), 2)
    
    # Mouth
    lip_color = (120, 120, 180) if gender_type == 'female' else (130, 150, 190)
    cv2.ellipse(img, (112, 150), (20, 10), 0, 0, 180, lip_color, 3)
    
    # Hair / features based on gender_type
    if gender_type == 'female':
        # Long hair arcs
        cv2.ellipse(img, (112, 90), (75, 75), 0, 180, 360, (30, 20, 10), -1)
        cv2.rectangle(img, (35, 90), (60, 180), (30, 20, 10), -1)
        cv2.rectangle(img, (164, 90), (189, 180), (30, 20, 10), -1)
    else:
        # Short hair cap
        cv2.ellipse(img, (112, 90), (70, 55), 0, 180, 360, (40, 35, 30), -1)
        
    return img

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
    categories = ['male', 'female']
    num_samples = 40
    
    print(f"Generating synthetic sample dataset in: {base_dir}")
    
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        
        for i in range(num_samples):
            img = generate_synthetic_face(cat, seed=i + (100 if cat == 'female' else 0))
            img_path = os.path.join(cat_dir, f"{cat}_sample_{i+1:03d}.jpg")
            cv2.imwrite(img_path, img)
            
    print(f"Successfully generated {num_samples} sample images for each class ('male', 'female').")

if __name__ == "__main__":
    main()
