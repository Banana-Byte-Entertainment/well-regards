import urllib.request
import tarfile
from pathlib import Path

def download_ssd_mobilenet_v2():
    """Download SSD MobileNet v2 COCO model - more compatible with OpenCV"""
    
    MODEL_URL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    MODEL_TAR = "ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
    MODEL_DIR = "ssd_mobilenet_v2_coco_2018_03_29"
    
    # Check if already downloaded
    if Path(MODEL_DIR).exists() and Path(f"{MODEL_DIR}/frozen_inference_graph.pb").exists():
        print(f"✓ Model already exists at {MODEL_DIR}")
        return MODEL_DIR
    
    print("Downloading SSD MobileNet v2 model...")
    print(f"URL: {MODEL_URL}")
    
    try:
        # Download
        urllib.request.urlretrieve(MODEL_URL, MODEL_TAR)
        print(f"✓ Downloaded {MODEL_TAR}")
        
        # Extract
        print("Extracting...")
        with tarfile.open(MODEL_TAR, 'r:gz') as tar:
            tar.extractall()
        print(f"✓ Extracted to {MODEL_DIR}")
        
        # Verify files
        pb_file = Path(f"{MODEL_DIR}/frozen_inference_graph.pb")
        if pb_file.exists():
            size_mb = pb_file.stat().st_size / (1024 * 1024)
            print(f"✓ Model file found: {size_mb:.2f} MB")
        else:
            print("✗ Model file not found after extraction!")
            return None
        
        # Clean up tar file
        Path(MODEL_TAR).unlink()
        print(f"✓ Cleaned up {MODEL_TAR}")
        
        return MODEL_DIR
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    model_dir = download_ssd_mobilenet_v2()
    
    if model_dir:
        print("\n" + "="*50)
        print("SUCCESS! Use these paths in your ImageReader:")
        print(f"  model_path='{model_dir}/frozen_inference_graph.pb'")
        print("="*50)
