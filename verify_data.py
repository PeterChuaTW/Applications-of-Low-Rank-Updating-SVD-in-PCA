"""
Verification script for ORL Face Database.

This script checks if the ORL database is correctly loaded and validates the data.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data.loader import load_orl_faces
import numpy as np


def verify_orl_database():
    """
    Verify that ORL Face Database is correctly loaded.
    """
    print("\n" + "=" * 60)
    print("ORL FACE DATABASE VERIFICATION")
    print("=" * 60 + "\n")
    
    # 載入數據（會自動下載如果不存在）
    faces, labels, is_real = load_orl_faces('data/ORL_Faces', auto_download=True)
    
    print("\n" + "=" * 60)
    print("VERIFICATION CHECKS")
    print("=" * 60)
    
    checks_passed = 0
    total_checks = 7
    
    # Check 1: 樣本數量
    try:
        assert faces.shape[0] == 400, f"Expected 400 faces, got {faces.shape[0]}"
        print(f"✓ Check 1/7: Total samples = {faces.shape[0]} ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 1/7: {e}")
    
    # Check 2: 特徵維度
    try:
        assert faces.shape[1] == 10304, f"Expected 10304 features, got {faces.shape[1]}"
        print(f"✓ Check 2/7: Feature dimension = {faces.shape[1]} (92×112 pixels) ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 2/7: {e}")
    
    # Check 3: 標籤範圍
    try:
        assert np.min(labels) == 0, f"Expected min label 0, got {np.min(labels)}"
        assert np.max(labels) == 39, f"Expected max label 39, got {np.max(labels)}"
        print(f"✓ Check 3/7: Label range = [{np.min(labels)}, {np.max(labels)}] (40 subjects) ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 3/7: {e}")
    
    # Check 4: 每個類別的樣本數
    try:
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 40, f"Expected 40 subjects, got {len(unique)}"
        assert np.all(counts == 10), f"Each subject should have 10 images, got {counts}"
        print(f"✓ Check 4/7: Subjects = {len(unique)}, Images per subject = {counts[0]} ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 4/7: {e}")
    
    # Check 5: 像素值範圍
    try:
        min_val, max_val = np.min(faces), np.max(faces)
        assert 0 <= min_val <= 255, f"Min pixel value {min_val} out of range"
        assert 0 <= max_val <= 255, f"Max pixel value {max_val} out of range"
        print(f"✓ Check 5/7: Pixel range = [{min_val:.2f}, {max_val:.2f}] ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 5/7: {e}")
    
    # Check 6: 數據類型
    try:
        assert faces.dtype == np.float64, f"Expected float64, got {faces.dtype}"
        print(f"✓ Check 6/7: Data type = {faces.dtype} ✓")
        checks_passed += 1
    except AssertionError as e:
        print(f"✗ Check 6/7: {e}")
    
    # Check 7: 真實 vs 合成數據
    if is_real:
        print(f"✓ Check 7/7: Using REAL ORL Face Database ✓")
        checks_passed += 1
    else:
        print(f"⚠ Check 7/7: Using SYNTHETIC data (not real ORL database)")
    
    # 總結
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULT: {checks_passed}/{total_checks} checks passed")
    print("=" * 60)
    
    if checks_passed == total_checks:
        print("\n🎉 All checks passed! Database is ready for experiments.")
        return True
    elif checks_passed >= 6:
        print("\n⚠ Most checks passed, but verify data source.")
        return True
    else:
        print("\n✗ Verification failed. Please check the database.")
        return False


if __name__ == "__main__":
    success = verify_orl_database()
    sys.exit(0 if success else 1)
