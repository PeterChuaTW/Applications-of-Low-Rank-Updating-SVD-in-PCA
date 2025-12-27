"""
Data loading utilities for ORL Face Database.
"""
import numpy as np
import os
from PIL import Image


def load_orl_faces(data_dir='data/ORL_Faces'):
    """
    Load ORL Face Database.
    
    The ORL database contains 400 images (40 subjects, 10 images each).
    Each image is 92x112 pixels (grayscale).
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the ORL face images
        Expected structure: data_dir/sX/Y.pgm where X is subject (1-40), Y is image (1-10)
        
    Returns:
    --------
    faces : array-like, shape (400, 10304)
        Face images flattened to vectors (92*112 = 10304)
    labels : array-like, shape (400,)
        Subject labels (0-39)
    """
    faces = []
    labels = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Warning: ORL Face Database directory '{data_dir}' not found.")
        print("Generating synthetic face data for demonstration...")
        return generate_synthetic_faces()
    
    # Load faces from directory structure
    for subject_id in range(1, 41):  # 40 subjects
        subject_dir = os.path.join(data_dir, f's{subject_id}')
        
        if not os.path.exists(subject_dir):
            print(f"Warning: Subject directory '{subject_dir}' not found.")
            continue
            
        for img_id in range(1, 11):  # 10 images per subject
            img_path = os.path.join(subject_dir, f'{img_id}.pgm')
            
            if not os.path.exists(img_path):
                # Try alternative naming conventions
                img_path = os.path.join(subject_dir, f'{img_id}.jpg')
                if not os.path.exists(img_path):
                    img_path = os.path.join(subject_dir, f'{img_id}.png')
            
            if os.path.exists(img_path):
                try:
                    # Load and convert to grayscale
                    img = Image.open(img_path).convert('L')
                    
                    # Resize to 92x112 if needed
                    if img.size != (92, 112):
                        img = img.resize((92, 112))
                    
                    # Convert to numpy array and flatten
                    face_vector = np.array(img).flatten().astype(np.float64)
                    
                    faces.append(face_vector)
                    labels.append(subject_id - 1)  # 0-indexed labels
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    if len(faces) == 0:
        print("No faces loaded. Generating synthetic face data...")
        return generate_synthetic_faces()
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    print(f"Loaded {len(faces)} face images from ORL database")
    return faces, labels


def generate_synthetic_faces(n_samples=400, height=92, width=112):
    """
    Generate synthetic face-like data for demonstration.
    
    Creates random images with some structure to simulate face data.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic faces to generate
    height : int
        Image height
    width : int
        Image width
        
    Returns:
    --------
    faces : array-like, shape (n_samples, height*width)
        Synthetic face images flattened to vectors
    labels : array-like, shape (n_samples,)
        Synthetic subject labels
    """
    print(f"Generating {n_samples} synthetic face images ({height}x{width} pixels)...")
    
    n_features = height * width
    faces = []
    labels = []
    
    n_subjects = 40
    images_per_subject = n_samples // n_subjects
    
    # Create synthetic faces with some structure
    for subject_id in range(n_subjects):
        # Each subject has a base pattern
        base_pattern = np.random.randn(n_features) * 50 + 128
        
        for img_id in range(images_per_subject):
            # Add variation to the base pattern
            variation = np.random.randn(n_features) * 20
            face = base_pattern + variation
            
            # Clip to valid pixel range
            face = np.clip(face, 0, 255)
            
            faces.append(face)
            labels.append(subject_id)
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    print(f"Generated {len(faces)} synthetic face images")
    return faces, labels


def normalize_faces(faces):
    """
    Normalize face data to zero mean and unit variance.
    
    Parameters:
    -----------
    faces : array-like, shape (n_samples, n_features)
        Face images as vectors
        
    Returns:
    --------
    normalized_faces : array-like, shape (n_samples, n_features)
        Normalized face images
    """
    mean = np.mean(faces, axis=0)
    std = np.std(faces, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    return (faces - mean) / std


def split_data(X, y, train_ratio=0.8, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data
    y : array-like, shape (n_samples,)
        Labels
    train_ratio : float
        Ratio of training data (0-1)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Split data
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def reshape_to_image(face_vector, height=92, width=112):
    """
    Reshape a flattened face vector back to image.
    
    Parameters:
    -----------
    face_vector : array-like, shape (height*width,)
        Flattened face image
    height : int
        Image height
    width : int
        Image width
        
    Returns:
    --------
    image : array-like, shape (height, width)
        Face image
    """
    return face_vector.reshape(height, width)
