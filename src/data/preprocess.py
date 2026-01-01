import numpy as np

def normalize_faces(faces):
    """
    Normalize face data to zero mean (Mean Centering for PCA).
    
    This is the standard preprocessing step for PCA.
    
    Parameters:
    -----------
    faces : array-like, shape (n_samples, n_features)
        Face images as vectors
        
    Returns:
    --------
    normalized_faces : array-like, shape (n_samples, n_features)
        Mean-centered face images
    mean_face : array-like, shape (n_features,)
        Mean face vector
    """
    mean_face = np.mean(faces, axis=0)
    centered_faces = faces - mean_face
    
    return centered_faces, mean_face