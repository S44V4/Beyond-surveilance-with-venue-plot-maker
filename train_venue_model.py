import os
import glob
import cv2
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np

from venue_graph_model import EndToEndVenueGraphModel
from crowd_twin.venue import VenueGraph, zone_masks, aggregate_perception_to_zones
from backend.balanced_ddpf import preprocess

class ShanghaiTechDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback if image is unreadable
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Preprocess using DDPF's preprocess function
        tensor, nw, nh = preprocess(img)
        # tensor is [1, 3, H, W]
        return tensor.squeeze(0) # [3, H, W]

def train_shanghaitech_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting training on {device}...")
    
    # Initialize the End-to-End Model
    model = EndToEndVenueGraphModel(device=device).to(device)
    
    # Freeze DDPF backbone to save memory and speed up training
    for param in model.ddpf.parameters():
        param.requires_grad = False
        
    model.train()
    model.ddpf.eval() # Keep DDPF in eval mode (BatchNorm, Dropout frozen)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)
    
    count_loss_fn = nn.SmoothL1Loss()
    risk_loss_fn = nn.CrossEntropyLoss()
    
    venue_path = "configs/venues/dronecrowd_analysis_grid.json"
    venue = VenueGraph.load(venue_path)
    
    # Prepare DDPF mask for pseudo-ground truth generation
    feature_size = (32, 32)
    masks_np = zone_masks(venue, feature_size[0], feature_size[1])
    
    # Load Dataset
    img_dir = r"C:\Users\USER\Downloads\beyond_surveillance_mvp\beyond_surveillance\ShanghaiTech\part_B\train_data\images"
    dataset = ShanghaiTechDataset(img_dir)
    # Using batch_size=1 since images are padded/resized but STRFE takes T sequence. 
    # We will use T=1.
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    epochs = 3
    print(f"Training on {len(dataset)} images for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_count_loss = 0.0
        epoch_risk_loss = 0.0
        
        for i, batch_images in enumerate(dataloader):
            # batch_images: [B, 3, 512, 512]
            batch_images = batch_images.to(device)
            B = batch_images.size(0)
            
            # Since our model expects [B, T, C, H, W], add T=1
            images_seq = batch_images.unsqueeze(1) # [B, 1, 3, 512, 512]
            
            # 1. Generate Pseudo-Ground Truth using DDPF
            with torch.no_grad():
                out = model.ddpf(batch_images) # [B, 3, 512, 512]
                den = torch.nn.functional.interpolate(out["density"], size=feature_size, mode="area")
                logits = torch.nn.functional.interpolate(out["heatmap_logits"], size=feature_size, mode="area")
                
                # Convert to numpy for aggregation
                den_np = den[0, 0].cpu().numpy()
                probs_np = (1 / (1 + np.exp(-np.clip(logits[0, 0].cpu().numpy(), -30, 30))))
                
                # Aggregate to get ground truth zone counts
                agg = aggregate_perception_to_zones(den_np, probs_np, masks_np, localization_threshold=0.25)
                gt_counts = torch.tensor(agg["count"], dtype=torch.float32, device=device).unsqueeze(0) # [1, N]
                
                # Generate pseudo risk labels based on count thresholds
                # 0=Safe (<5), 1=Watch (<15), 2=High (<30), 3=Critical (>=30)
                gt_risks = torch.zeros_like(gt_counts, dtype=torch.long)
                gt_risks[gt_counts >= 5] = 1
                gt_risks[gt_counts >= 15] = 2
                gt_risks[gt_counts >= 30] = 3

            optimizer.zero_grad()
            
            # 2. Forward pass through the rest of the model (STRFE + Graph)
            outputs = model(images_seq, venue)
            
            # 3. Calculate Loss
            # current_zone_state is [B, N, 7] -> count is index 0
            pred_counts = outputs["current_zone_state"][:, :, 0]
            loss_count = count_loss_fn(pred_counts, gt_counts)
            
            # risk_logits is [B, N, 4]
            pred_risks = outputs["risk_logits"].view(-1, 4)
            loss_risk = risk_loss_fn(pred_risks, gt_risks.view(-1))
            
            loss = loss_count + 0.5 * loss_risk
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_count_loss += loss_count.item()
            epoch_risk_loss += loss_risk.item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}] - "
                      f"Count Loss: {loss_count.item():.4f}, Risk Loss: {loss_risk.item():.4f}")
                
        print(f"==== Epoch {epoch+1} Summary ====")
        print(f"Avg Count Loss: {epoch_count_loss/len(dataloader):.4f}")
        print(f"Avg Risk Loss: {epoch_risk_loss/len(dataloader):.4f}")

    print("Training finished! Saving model weights...")
    torch.save(model.state_dict(), "shanghaitech_venue_model.pt")
    print("Saved to shanghaitech_venue_model.pt")

if __name__ == "__main__":
    train_shanghaitech_loop()
