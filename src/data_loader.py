"""
Data loading utilities for ORL Face Database with automatic download.
"""
import numpy as np
import os
from PIL import Image
import urllib.request
import zipfile
import shutil


def download_orl_database(data_dir='data/ORL_Faces'):
    """
    Automatically download and extract ORL Face Database.
    
    Downloads from the official AT&T (formerly ORL) database archive.
    
    Parameters:
    -----------
    data_dir : str
        Target directory for the database
        
    Returns:
    --------
    success : bool
        True if download successful, False otherwise
    """
    # 如果目錄已存在，不需要下載
    if os.path.exists(data_dir):
        print(f"✓ ORL database already exists at {data_dir}")
        return True
    
    # 官方下載連結
    url = 'https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip'
    zip_path = 'data/att_faces.zip'
    temp_extract_dir = 'data/att_faces_temp'
    
    print("=" * 60)
    print("Downloading ORL Face Database...")
    print(f"Source: {url}")
    print("=" * 60)
    
    try:
        # 確保 data 目錄存在
        os.makedirs('data', exist_ok=True)
        
        # 下載 ZIP 檔案
        print("Downloading... (this may take a minute)")
        urllib.request.urlretrieve(url, zip_path)
        print(f"✓ Downloaded to {zip_path}")
        
        # 解壓縮到臨時目錄
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
        print("✓ Extracted successfully")
        
        # 找到解壓後的實際目錄（可能是 orl_faces 或 att_faces）
        extracted_dirs = os.listdir(temp_extract_dir)
        if extracted_dirs:
            source_dir = os.path.join(temp_extract_dir, extracted_dirs[0])
            # 移動到目標位置
            shutil.move(source_dir, data_dir)
            print(f"✓ Moved to {data_dir}")
        else:
            # 如果直接解壓到根目錄
            shutil.move(temp_extract_dir, data_dir)
            print(f"✓ Moved to {data_dir}")
        
        # 清理臨時檔案
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        print("✓ Cleaned up temporary files")
        
        # 驗證下載結果
        if verify_database_structure(data_dir):
            print("=" * 60)
            print("✓ ORL Face Database ready!")
            print("=" * 60)
            return True
        else:
            print("⚠ Warning: Database structure may be incorrect")
            return False
            
    except Exception as e:
        print(f"✗ Failed to download ORL database: {e}")
        print("Cleaning up partial downloads...")
        
        # 清理失敗的下載
        for path in [zip_path, temp_extract_dir, data_dir]:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        
        print("=" * 60)
        print("Will use synthetic data instead.")
        print("=" * 60)
        return False


def verify_database_structure(data_dir='data/ORL_Faces'):
    """
    Verify that the ORL database has the correct structure.
    
    Expected: 40 subject directories (s1-s40), each with 10 images.
    
    Parameters:
    -----------
    data_dir : str
        Database directory to verify
        
    Returns:
    --------
    valid : bool
        True if structure is valid
    """
    if not os.path.exists(data_dir):
        return False
    
    # 檢查是否有 s1 到 s40 的目錄
    subject_count = 0
    for i in range(1, 41):
        subject_dir = os.path.join(data_dir, f's{i}')
        if os.path.exists(subject_dir):
            subject_count += 1
    
    return subject_count >= 30  # 至少要有 30 個主題目錄


def load_orl_faces(data_dir='data/ORL_Faces', auto_download=True):
    """
    Load ORL Face Database with automatic download option.
    
    The ORL database contains 400 images (40 subjects, 10 images each).
    Each image is 92x112 pixels (grayscale).
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the ORL face images
        Expected structure: data_dir/sX/Y.pgm where X is subject (1-40), Y is image (1-10)
    auto_download : bool
        If True, automatically download the database if not found
        
    Returns:
    --------
    faces : array-like, shape (400, 10304)
        Face images flattened to vectors (92*112 = 10304)
    labels : array-like, shape (400,)
        Subject labels (0-39)
    is_real_data : bool
        True if using real ORL data, False if using synthetic data
    """
    # 嘗試自動下載
    if not os.path.exists(data_dir) and auto_download:
        print("ORL Face Database not found. Attempting automatic download...")
        download_success = download_orl_database(data_dir)
        if not download_success:
            print("Using synthetic face data for demonstration...")
            return *generate_synthetic_faces(), False
    
    # 檢查目錄是否存在
    if not os.path.exists(data_dir):
        print(f"Warning: ORL Face Database directory '{data_dir}' not found.")
        print("Generating synthetic face data for demonstration...")
        return *generate_synthetic_faces(), False
    
    faces = []
    labels = []
    
    # 載入真實的人臉影像
    print(f"Loading ORL Face Database from {data_dir}...")
    for subject_id in range(1, 41):  # 40 subjects
        subject_dir = os.path.join(data_dir, f's{subject_id}')
        
        if not os.path.exists(subject_dir):
            continue
            
        for img_id in range(1, 11):  # 10 images per subject
            img_path = os.path.join(subject_dir, f'{img_id}.pgm')
            
            # 嘗試其他格式
            if not os.path.exists(img_path):
                for ext in ['.jpg', '.png', '.gif', '.bmp']:
                    alt_path = os.path.join(subject_dir, f'{img_id}{ext}')
                    if os.path.exists(alt_path):
                        img_path = alt_path
                        break
            
            if os.path.exists(img_path):
                try:
                    # 載入並轉換為灰階
                    img = Image.open(img_path).convert('L')
                    
                    # 調整大小至 92x112（如果需要）
                    if img.size != (92, 112):
                        img = img.resize((92, 112))
                    
                    # 轉換為 numpy 陣列並展平
                    face_vector = np.array(img).flatten().astype(np.float64)
                    
                    faces.append(face_vector)
                    labels.append(subject_id - 1)  # 0-indexed labels
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    # 如果沒有載入任何影像，使用合成數據
    if len(faces) == 0:
        print("No faces loaded from database. Generating synthetic face data...")
        return *generate_synthetic_faces(), False
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    # 印出統計資訊
    print("=" * 60)
    print(f"✓ Successfully loaded {len(faces)} real face images")
    print(f"  - Subjects: {len(np.unique(labels))}")
    print(f"  - Images per subject: ~{len(faces) // len(np.unique(labels))}")
    print(f"  - Image dimensions: 92×112 pixels (10304 features)")
    print("=" * 60)
    
    return faces, labels, True


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
    print("=" * 60)
    print(f"Generating {n_samples} synthetic face images...")
    print(f"Image size: {height}×{width} pixels")
    print("=" * 60)
    
    n_features = height * width
    faces = []
    labels = []
    
    n_subjects = 40
    images_per_subject = n_samples // n_subjects
    
    # 創建具有結構的合成人臉
    for subject_id in range(n_subjects):
        # 每個主題有一個基礎模式
        base_pattern = np.random.randn(n_features) * 50 + 128
        
        for img_id in range(images_per_subject):
            # 加入變異
            variation = np.random.randn(n_features) * 20
            face = base_pattern + variation
            
            # 限制在有效的像素範圍
            face = np.clip(face, 0, 255)
            
            faces.append(face)
            labels.append(subject_id)
    
    faces = np.array(faces)
    labels = np.array(labels)
    
    print(f"✓ Generated {len(faces)} synthetic face images")
    print("⚠ Note: Using synthetic data, not real ORL database")
    print("=" * 60)
    
    return faces, labels


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
