import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Add the local directory to the path so we can import backend and crowd_twin
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.balanced_ddpf import BalancedPredictor
from crowd_twin.venue import VenueGraph, zone_masks, aggregate_perception_to_zones

def main():
    print("Loading Predictor...")
    # It hardcodes model path to store.ROOT / "models/balanced_ddpf.pt"
    # Ensure it exists (we copied it earlier)
    try:
        predictor = BalancedPredictor()
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return

    img_path = r"C:\Users\USER\Downloads\beyond_surveillance_mvp\beyond_surveillance\ShanghaiTech\part_B\train_data\images\IMG_1.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return
        
    print(f"Reading image {img_path}")
    img = cv2.imread(img_path)
    
    # Run the model
    print("Running DDPF model...")
    # The image is scaled inside export_sequence to 512x512
    # but the output sizes might vary. Let's see.
    result = predictor.export_sequence([img], sequence_id="test", fps=1.0)
    
    # Extract density and logits
    density = result.density[0, 0] # shape (H, W)
    logits = result.localization_logits[0, 0] # shape (H, W)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    
    print("Loading VenueGraph...")
    venue_path = "configs/venues/dronecrowd_analysis_grid.json"
    venue = VenueGraph.load(venue_path)
    
    print("Generating zone masks...")
    # The output density map has size `feature_size`, which is (32, 32).
    # But wait, `export_sequence` resizes valid density to `feature_size[::-1]`?
    # Let's check `density.shape`
    print(f"Density shape: {density.shape}, Probabilities shape: {probabilities.shape}")
    
    # We generate masks at the size of the density map
    masks = zone_masks(venue, density.shape[0], density.shape[1])
    
    print("Aggregating perception to zones...")
    agg = aggregate_perception_to_zones(density, probabilities, masks, localization_threshold=0.25)
    
    counts = agg["count"]
    densities = agg["density"]
    
    print("Plotting...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Draw original image as background? The shapes are different, but let's draw polygons over a blank canvas or the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb, extent=[0, venue.width, venue.height, 0])
    
    # The venue is defined in coordinates [0, venue.width] and [0, venue.height]
    import matplotlib.patches as patches
    
    max_count = max(counts) if max(counts) > 0 else 1.0
    
    for i, zone in enumerate(venue.zones):
        poly = np.array(zone.polygon)
        count = counts[i]
        
        # Color based on count
        intensity = count / max_count
        color = plt.cm.jet(intensity)
        
        patch = patches.Polygon(poly, closed=True, facecolor=color, alpha=0.5, edgecolor='black')
        ax.add_patch(patch)
        
        cx, cy = zone.centroid
        ax.text(cx, cy, f"{zone.zone_id}\nCount: {count:.1f}", color='white', 
                ha='center', va='center', fontsize=8, weight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
                
    ax.set_title("Venue Plot - Crowd Count per Zone")
    ax.set_xlim(0, venue.width)
    ax.set_ylim(venue.height, 0) # Invert y axis for image coordinates
    
    out_file = "venue_plot.jpg"
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    print(f"Saved venue plot to {out_file}")

if __name__ == "__main__":
    main()
