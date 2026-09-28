import torch
import cv2
import numpy as np
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class BEVMapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Depth-Anything Model on {device}...")
        # Load Depth Anything V2/3 model from HuggingFace
        self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=0 if device=="cuda" else -1)
        
    def generate_bev(self, image_bgr, pitch_deg=30):
        # 1. Estimate Depth
        from PIL import Image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        # Note: pipeline takes the image directly
        depth_output = self.depth_estimator(image_pil)
        
        # Depth is usually a PIL image, convert to numpy
        depth_map = np.array(depth_output["depth"])
        
        # Normalize depth to a reasonable relative scale
        depth_map = cv2.resize(depth_map, (image_bgr.shape[1], image_bgr.shape[0]))
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        
        # DepthAnything returns inverse depth (disparity). So closer objects have higher values.
        # We invert it to get actual relative depth:
        Z = 1.0 / (depth_map + 0.1) 
        
        H, W = image_bgr.shape[:2]
        
        # 2. Camera Intrinsics (Assume 60 deg FOV)
        fov = 60 * np.pi / 180
        fx = W / (2 * np.tan(fov / 2))
        fy = fx
        cx = W / 2
        cy = H / 2
        
        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # 3. Project to 3D Camera Coordinates
        X_c = (u - cx) * Z / fx
        Y_c = (v - cy) * Z / fy
        Z_c = Z
        
        # 4. Rotate to World Coordinates based on Camera Pitch
        # Camera is pitched down by `pitch_deg` degrees.
        theta = np.radians(pitch_deg)
        
        # Rotation matrix around X-axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        
        points_c = np.stack([X_c, Y_c, Z_c], axis=-1).reshape(-1, 3)
        points_w = points_c @ R_x.T # Matrix multiplication
        
        X_w = points_w[:, 0]
        Y_w = points_w[:, 1] # Up/Down (Height)
        Z_w = points_w[:, 2] # Forward/Backward (Depth along ground)
        
        # 5. Render Top-Down BEV (Bird's Eye View)
        # We want to plot X_w vs Z_w. We map them to a 2D image grid.
        bev_res = 500
        
        # Determine bounds
        min_x, max_x = np.percentile(X_w, 2), np.percentile(X_w, 98)
        min_z, max_z = np.percentile(Z_w, 2), np.percentile(Z_w, 98)
        
        bev_image = np.zeros((bev_res, bev_res, 3), dtype=np.uint8)
        
        # Filter points within bounds
        valid = (X_w > min_x) & (X_w < max_x) & (Z_w > min_z) & (Z_w < max_z)
        X_valid = X_w[valid]
        Z_valid = Z_w[valid]
        
        colors = image_rgb.reshape(-1, 3)[valid]
        
        # Map to BEV pixels
        # X maps to width (x-axis), Z maps to height (y-axis)
        # We invert Z so further away is top of image
        x_bev = ((X_valid - min_x) / (max_x - min_x) * (bev_res - 1)).astype(np.int32)
        y_bev = bev_res - 1 - ((Z_valid - min_z) / (max_z - min_z) * (bev_res - 1)).astype(np.int32)
        
        # Splat points into image
        for x, y, c in zip(x_bev, y_bev, colors):
            bev_image[y, x] = c
            
        # Dilate to fill holes in the point cloud splat
        kernel = np.ones((3, 3), np.uint8)
        bev_image = cv2.dilate(bev_image, kernel, iterations=1)
        
        # Convert back to BGR for saving
        bev_bgr = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)
        return bev_bgr, depth_map

def test_bev():
    mapper = BEVMapper()
    img_path = r"C:\Users\USER\Downloads\beyond_surveillance_mvp\beyond_surveillance\ShanghaiTech\part_B\train_data\images\IMG_1.jpg"
    img = cv2.imread(img_path)
    if img is not None:
        bev_img, depth = mapper.generate_bev(img, pitch_deg=45)
        cv2.imwrite("bev_floorplan.jpg", bev_img)
        print("Saved flat top-down floorplan to bev_floorplan.jpg")
    else:
        print("Image not found")

if __name__ == "__main__":
    test_bev()
