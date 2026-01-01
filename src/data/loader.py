"""
Data loading utilities for ORL Face Database with automatic download.
"""
import numpy as np
import os
from PIL import Image
import urllib.request
import zipfile
import shutil
import subprocess
import sys


def install_gdown():
    """Install gdown if not available."""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown for Google Drive download...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
            return True
        except:
            return False


def download_from_google_drive(data_dir='data/ORL_Faces'):
    """
    Download ORL Database from Google Drive using gdown.
    
    This is a backup method if the official download fails.
    
    Parameters:
    -----------
    data_dir : str
        Target directory for the database
        
    Returns:
    --------
    success : bool
        True if download successful
    """
    print("="*60)
    print("Downloading from Google Drive (backup method)...")
    print("="*60)
    
    # Google Drive folder URL (public)
    drive_url = "https://drive.google.com/drive/folders/1c3cOMdfy0jkCTWHFhIesLCKb9t57rywa"
    
    try:
        # Install gdown if needed
        if not install_gdown():
            print("✗ Failed to install gdown")
            return False
        
        import gdown
        
        # Download folder
        print(f"Downloading from: {drive_url}")
        print("This may take a few minutes...")
        
        # Ensure parent directory exists
        os.makedirs('data', exist_ok=True)
        
        # Download folder
        gdown.download_folder(drive_url, output=data_dir, quiet=False, use_cookies=False)
        
        if os.path.exists(data_dir) and verify_database_structure(data_dir):
            print("✓ Downloaded successfully from Google Drive")
            return True
        else:
            print("✗ Download completed but structure verification failed")
            return False
            
    except Exception as e:
        print(f"✗ Google Drive download failed: {e}")
        return False


def download_orl_database(data_dir='data/ORL_Faces'):
    """
    Automatically download and extract ORL Face Database.
    
    Tries official source first, then Google Drive backup.
    
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
    if os.path.exists(data_dir) and verify_database_structure(data_dir):
        print(f"✓ ORL database already exists at {data_dir}")
        return True
    
    # 方法 1: 官方下載連結
    url = 'https://www.cl.cam.ac.uk/Research/DTG/attarchive/pub/data/att_faces.zip'
    zip_path = 'data/att_faces.zip'
    temp_extract_dir = 'data/att_faces_temp'
    
    print("="*60)
    print("Method 1: Downloading from official source...")
    print(f"Source: {url}")
    print("="*60)
    
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
        
        # 找到解壓後的實際目錄
        extracted_items = os.listdir(temp_extract_dir)
        
        # 檢查是否直接包含 s1, s2, ... 資料夾
        has_subject_folders = any(item.startswith('s') and item[1:].isdigit() for item in extracted_items)
        
        if has_subject_folders:
            # 直接就是目標資料夾結構
            shutil.move(temp_extract_dir, data_dir)
            print(f"✓ Moved to {data_dir}")
        elif len(extracted_items) == 1:
            # 有一層子資料夾
            source_dir = os.path.join(temp_extract_dir, extracted_items[0])
            shutil.move(source_dir, data_dir)
            shutil.rmtree(temp_extract_dir)
            print(f"✓ Moved to {data_dir}")
        else:
            print("⚠ Warning: Unexpected directory structure")
            return False
        
        # 清理臨時檔案
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(temp_extract_dir):
            shutil.rmtree(temp_extract_dir)
        print("✓ Cleaned up temporary files")
        
        # 驗證下載結果
        if verify_database_structure(data_dir):
            print("="*60)
            print("✓ ORL Face Database ready!")
            print("="*60)
            return True
        else:
            print("⚠ Warning: Database structure may be incorrect")
            # 清理失敗的下載
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            # 嘗試 Google Drive 備用下載
            return download_from_google_drive(data_dir)
            
    except Exception as e:
        print(f"✗ Official download failed: {e}")
        print("")
        
        # 清理失敗的下載
        for path in [zip_path, temp_extract_dir, data_dir]:
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
        
        # 方法 2: Google Drive 備用下載
        print("Trying backup download from Google Drive...")
        return download_from_google_drive(data_dir)


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
        if os.path.exists(subject_dir) and os.path.isdir(subject_dir):
            subject_count += 1
    
    return subject_count >= 30  # 至少要有 30 個主題目錄


def load_orl_faces(data_dir='data/ORL_Faces', auto_download=True):
    """
    Load ORL Face Database with automatic download option.
    
    The ORL database contains 400 images (40 subjects, 10 images each).
    Each image is 92x112 pixels (width x height, grayscale).
    
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
            print("")
            print("All download methods failed.")
            print("Using synthetic face data for demonstration...")
            print("")
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
                    # PIL.Image.size is (width, height)
                    if img.size != (92, 112):
                        img = img.resize((92, 112))
                    
                    # 轉換為 numpy 陣列並展平
                    # NumPy array from PIL is (height, width) = (112, 92)
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
    print("="*60)
    print(f"✓ Successfully loaded {len(faces)} real face images")
    print(f"  - Subjects: {len(np.unique(labels))}")
    print(f"  - Images per subject: ~{len(faces) // len(np.unique(labels))}")
    print(f"  - Image dimensions: 112 (H) × 92 (W) pixels (10304 features)")
    print("="*60)
    
    return faces, labels, True


def generate_synthetic_faces(n_samples=400, height=112, width=92):
    """
    Generate synthetic face-like data for demonstration.
    
    Creates random images with some structure to simulate face data.
    
    Parameters:
    -----------
    n_samples : int
        Number of synthetic faces to generate
    height : int
        Image height (ORL standard: 112)
    width : int
        Image width (ORL standard: 92)
        
    Returns:
    --------
    faces : array-like, shape (n_samples, height*width)
        Synthetic face images flattened to vectors
    labels : array-like, shape (n_samples,)
        Synthetic subject labels
    """
    print("="*60)
    print(f"Generating {n_samples} synthetic face images...")
    print(f"Image size: {height} (H) × {width} (W) pixels")
    print("="*60)
    
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
    print("="*60)
    
    return faces, labels





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


def reshape_to_image(face_vector, height=112, width=92):
    """
    Reshape a flattened face vector back to image.
    
    IMPORTANT: ORL Face Database standard dimensions are:
    - Height: 112 pixels (rows)
    - Width: 92 pixels (columns)
    
    Parameters:
    -----------
    face_vector : array-like, shape (height*width,)
        Flattened face image (default: 10304 = 112*92)
    height : int
        Image height in pixels (default: 112 for ORL)
    width : int
        Image width in pixels (default: 92 for ORL)
        
    Returns:
    --------
    image : array-like, shape (height, width)
        Face image as 2D array
    """
    return face_vector.reshape(height, width)
