# %% [markdown]
# # Vision Transformer Attention Heatmap Pipeline
# This script loads an fine-tuned Vision Transformer (ViT) model,
# processes Chest X-Ray images, predicts Tuberculosis (TB) presence,
# and generates an attention-based heatmap indicating the affected region.

# %%
# ==========================================
# Core Pipeline Functions
# ==========================================
import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import ViTForImageClassification

# Setup Execution Device (Use GPU if available for faster inference, fallback to CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_class_mapping(mapping_path):
    """
    Loads class-to-index mapping from a JSON file.
    Returns a dictionary mapping indices to class labels.
    """
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Ensure mapping format is idx -> class name (e.g., {0: "health", 1: "tb"})
    if all(isinstance(v, int) for v in class_mapping.values()):
        idx_to_class = {v: k for k, v in class_mapping.items()}
    else:
        # If it's already id -> class, cast keys to integers
        idx_to_class = {int(k): v for k, v in class_mapping.items()}
        
    return idx_to_class

def load_model(weights_path, num_labels=2):
    """
    Loads the HuggingFace ViT model and customized classification head.
    Base model: google/vit-base-patch16-224 (requires 224x224 input images).
    """
    # Initialize base ViT model
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=num_labels,
        ignore_mismatched_sizes=True, # Allows overriding output classification head
        output_attentions=True        # Essential for extracting heatmap attention
    )
    
    # Load Fine-tuned Weights
    state_dict = torch.load(weights_path, map_location=device)
    
    # Account for various default saving formats
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval() # Set to evaluation mode for inference
    
    return model

def preprocess_image(image_path):
    """
    Loads an image (.jpg or .png), resizes it to 224x224, and normalizes it.
    Follows required ImageNet standards for standard ViT inference.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Define Image Preprocessing Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    image_np = np.array(image)
    
    return input_tensor, image_np

def predict(model, input_tensor):
    """
    Runs model inference without gradients and extracts prediction & confidence score.
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        
        # Apply Softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1)
        
        # Identify highest probability index
        conf, pred_idx = torch.max(probs, dim=1)
        
    return pred_idx.item(), conf.item(), outputs

def extract_attention(outputs):
    """
    Extracts the self-attention weights from the final ViT layer to identify 
    where the model was "looking" during classification. 
    """
    # Extract last layer attentions
    # Shape: (batch_size, num_heads, sequence_length, sequence_length)
    attentions = outputs.attentions[-1] 
    
    # Average attention across all multiple attention heads
    attentions_mean = torch.mean(attentions, dim=1)
    
    # Retrieve attention from the CLS token (index 0) to all image patches (indices 1 to end)
    cls_attention = attentions_mean[0, 0, 1:]
    
    # Reshape total patch sequence length back to 2D spatial grid (14x14)
    # 224x224 resolution with 16x16 patch sizes = 14x14 grid (196 total patches)
    grid_size = int(np.sqrt(cls_attention.size(0)))
    cls_attention = cls_attention.reshape(grid_size, grid_size)
    
    # Normalize attention grid between [0, 1] for visual plotting
    cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
    
    return cls_attention.cpu().numpy()

def generate_heatmap(image_np, attention_map, save_path=None):
    """
    Upsamples 14x14 attention grid maps to overlay cleanly onto the original 
    standard dimensions using OpenCV.
    """
    h, w, _ = image_np.shape
    
    # Upsample map
    attention_resized = cv2.resize(attention_map, (w, h))
    
    # Generate colormap display
    heatmap = np.uint8(255 * attention_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Combine original RGB picture data and colored heatmap arrays
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)
    
    # Bring back to RGB for matplotlib / Pillow compatibility
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    if save_path:
        # Note: Depending on path type, plt.imsave inherently expects RGB
        plt.imsave(save_path, overlay_rgb)
        
    return overlay_rgb

def get_region(attention_map):
    """
    Separates attention spatial layout map into 6 regions.
    Returns textual representation of heaviest attended region.
    """
    grid_h, grid_w = attention_map.shape
    
    # Demarcate spatial split indices
    col_mid = grid_w // 2        # Left / Right Split
    row_top = grid_h // 3        # Upper Limit Divider
    row_bot = 2 * (grid_h // 3)  # Lower Limit Divider
    
    # Aggregate attention totals
    regions = {
        "upper left": attention_map[:row_top, :col_mid].mean(),
        "upper right": attention_map[:row_top, col_mid:].mean(),
        "middle left": attention_map[row_top:row_bot, :col_mid].mean(),
        "middle right": attention_map[row_top:row_bot, col_mid:].mean(),
        "lower left": attention_map[row_bot:, :col_mid].mean(),
        "lower right": attention_map[row_bot:, col_mid:].mean(),
    }
    
    # Isolate maximum attentive region
    best_region = max(regions.items(), key=lambda x: x[1])[0]
    return f"{best_region} lung"


# %%
# ==========================================
# Testing Module Cell (Data Sampling & Visualization)
# ==========================================
import kagglehub
import glob
import random

def run_pipeline_test():
    """
    Executable isolated test wrapper integrating directly with 'tirthachhetri/tuberclosis-dataset' on Kaggle.
    Requires minimum of 5 + 5 images locally verified cache access.
    """
    print("\n--- Model Final Architecture Test Pipeline Initiated ---")
    
    # 1. Download Kaggle Dataset (Automatically caches dataset to local machine)
    print("Downloading/Locating Kaggle dataset...")
    path = kagglehub.dataset_download("tirthachhetri/tuberclosis-dataset")
    print(f"Path to dataset cached source: {path}")
    
    health_dir = os.path.join(path, "TB_Split", "valid", "health")
    tb_dir = os.path.join(path, "TB_Split", "valid", "tb")
    
    # Identify images (.png or .jpg natively accepted)
    def get_images(folder_path):
        return glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        
    # Pick randomly from distributions
    health_imgs = random.sample(get_images(health_dir), 5)
    tb_imgs = random.sample(get_images(tb_dir), 5)
    
    test_images = health_imgs + tb_imgs
    random.shuffle(test_images) # Shuffle batches for randomization sequence
    
    # Configure references setup mapping folder paths
    base_folder = 'mini-folder'
    model_path = os.path.join(base_folder, 'vit_tb_model.pth')
    mapping_path = os.path.join(base_folder, 'class_mapping.json')
    
    # Validate execution presence
    if not os.path.exists(model_path) or not os.path.exists(mapping_path):
        print(f"\n[WARNING] Directory dependencies missing!")
        print(f"Cannot proceed. Ensure '{base_folder}' folder contains:")
        print(f" - {os.path.basename(model_path)}\n - {os.path.basename(mapping_path)}")
        return
        
    # 2. Pipeline Initialization
    print("\nInitializing Vision Transformer Weights...")
    idx_to_class = load_class_mapping(mapping_path)
    model = load_model(model_path, num_labels=len(idx_to_class))
    
    # 3. Execution Output Display
    for i, img_path in enumerate(test_images):
        print(f"\n[{i+1}/10] Testing Image: {os.path.basename(img_path)}")
        
        # Preprocessing Extractor
        input_tensor, original_image_np = preprocess_image(img_path)
        
        # Predict Class Vector
        pred_idx, conf, outputs = predict(model, input_tensor)
        predicted_class = idx_to_class.get(pred_idx, f"Class_{pred_idx}")
        
        # Attention Mapping Analysis Extraction
        attention_map = extract_attention(outputs)
        top_region = get_region(attention_map)
        
        # Rendering Output Processing
        save_heatmap_path = f"heatmap_output_{i+1}.png"
        heatmap_overlay = generate_heatmap(original_image_np, attention_map, save_path=save_heatmap_path)
        
        # Formatted Return Standard Out
        print(f" > Prediction:       {predicted_class}")
        print(f" > Confidence score: {conf:.4f}")
        print(f" > Attention Region: {top_region}")
        print(f" > Saved Heatmap:    {save_heatmap_path}")
        
        # Standard matplotlib subplot comparison graph
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_image_np)
        axes[0].set_title("Original Chest X-Ray")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_overlay)
        axes[1].set_title(f"Attention Heatmap Overlay")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_pipeline_test()
