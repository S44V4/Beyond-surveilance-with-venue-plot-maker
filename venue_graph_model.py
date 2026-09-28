import torch
import torch.nn as nn
from backend.balanced_ddpf import DDPFNetLastHope
from crowd_twin.models.strfe import STRFE
from crowd_twin.models.graph_reasoner import GraphRiskReasoner
from crowd_twin.venue import VenueGraph, zone_masks, node_static_features
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EndToEndVenueGraphModel(nn.Module):
    """
    An end-to-end PyTorch Neural Network that takes raw images,
    processes them through DDPF (Perception), STRFE (Temporal Reasoning),
    and GraphRiskReasoner (Venue Graph GNN) to predict zone-level crowd states and risks.
    """
    def __init__(
        self,
        strfe_input_channels=256,
        strfe_feature_dim=128,
        strfe_horizons=(5, 15, 30),
        graph_hidden_dim=128,
        graph_heads=4,
        graph_layers=2,
        risk_classes=4,
        device="cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.horizons = strfe_horizons
        
        # 1. Perception: DDPF
        self.ddpf = DDPFNetLastHope()
        
        # 2. Temporal Feature Extraction: STRFE
        self.strfe = STRFE(
            input_channels=strfe_input_channels,
            feature_dim=strfe_feature_dim,
            horizons_seconds=self.horizons,
            temporal_layers=1,
            use_mamba=False
        )
        
        # 3. Spatial/Venue Graph Reasoning: GraphRiskReasoner
        self.graph = GraphRiskReasoner(
            feature_dim=strfe_feature_dim,
            hidden_dim=graph_hidden_dim,
            static_feature_dim=4, # From node_static_features default
            graph_heads=graph_heads,
            graph_layers=graph_layers,
            risk_classes=risk_classes,
            horizons_seconds=self.horizons
        )

    def forward(self, images, venue_graph, image_size=(512, 512), feature_size=(32, 32)):
        """
        Forward pass.
        images: [B, T, C, H, W] - a sequence of images (RGB, normalized).
        venue_graph: A VenueGraph object defining the topology.
        """
        B, T, C, H, W = images.shape
        
        # We need the zone masks for STRFE pooling
        masks = zone_masks(venue_graph, feature_size[0], feature_size[1])
        masks_tensor = torch.from_numpy(masks).to(self.device).float()
        
        # Static features & Adjacency for Graph
        static_feats = node_static_features(venue_graph, feature_dim=4)
        static_tensor = torch.from_numpy(static_feats).to(self.device).float().unsqueeze(0).expand(B, -1, -1)
        adj_matrix = np.array(venue_graph.adjacency, dtype=bool)
        adj_tensor = torch.from_numpy(adj_matrix).to(self.device).unsqueeze(0).expand(B, -1, -1)
        
        visual_features = []
        densities = []
        
        # 1. Pass through DDPF (process frame by frame)
        for b in range(B):
            b_feats = []
            b_densities = []
            for t in range(T):
                # DDPF output
                out = self.ddpf(images[b, t].unsqueeze(0))
                # features: [1, 256, 32, 32]
                b_feats.append(out["features"].squeeze(0))
                # density: [1, 1, 512, 512] -> resize to feature size
                den = torch.nn.functional.interpolate(
                    out["density"], size=feature_size, mode="area"
                )
                b_densities.append(den.squeeze(0))
                
            visual_features.append(torch.stack(b_feats))
            densities.append(torch.stack(b_densities))
            
        visual_features = torch.stack(visual_features) # [B, T, C, Hf, Wf]
        densities = torch.stack(densities)             # [B, T, 1, Hf, Wf]
        
        # Mock optical flow (zeros) for this pipeline if not computed
        flow = torch.zeros(B, T, 4, feature_size[0], feature_size[1], device=self.device)
        
        # 2. Pass through STRFE
        strfe_out = self.strfe(
            visual_features=visual_features,
            density=densities,
            localization_logits=densities, # reusing for signature match if logits needed
            flow=flow,
            zone_masks=masks_tensor,
            sensors=None,
            sensor_mask=None
        )
        
        # 3. Pass through GraphRiskReasoner
        graph_out = self.graph(
            current_zone_features=strfe_out["current_zone_features"],
            future_zone_features=strfe_out["future_zone_features"],
            adjacency=adj_tensor,
            node_static=static_tensor,
            current_state_prior=strfe_out["current_zone_state"],
            future_state_prior=strfe_out["future_zone_state"]
        )
        
        return graph_out

    def predict_and_plot(self, image_np, venue_path, out_file="end_to_end_venue_plot.jpg"):
        """
        Convenience method to take a single image numpy array (BGR), 
        format it for the model, run inference (T=1), and output a venue plot.
        """
        # Load venue
        venue = VenueGraph.load(venue_path)
        
        # Preprocess image
        from backend.balanced_ddpf import preprocess
        tensor, nw, nh = preprocess(image_np)
        
        # Add batch and time dimensions: [1, 1, 3, H, W]
        images = tensor.unsqueeze(0).to(self.device)
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, venue)
            
        # Extract graph outputs for plotting
        # Let's plot the predicted current state's count (index 0 of state features)
        current_state = outputs["current_zone_state"][0].cpu().numpy() # [N, 7]
        # state features order: count, density, velocity, acceleration, divergence, congestion, risk
        counts = current_state[:, 0]
        
        risk_logits = outputs["risk_logits"][0].cpu().numpy() # [N, classes]
        predicted_risks = risk_logits.argmax(axis=-1)
        
        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb, extent=[0, venue.width, venue.height, 0])
        
        import matplotlib.patches as patches
        max_count = max(counts) if max(counts) > 0 else 1.0
        
        risk_colors = ['green', 'yellow', 'orange', 'red']
        
        for i, zone in enumerate(venue.zones):
            poly = np.array(zone.polygon)
            count = counts[i]
            risk = predicted_risks[i]
            
            # Color by predicted risk or count
            color = risk_colors[risk % len(risk_colors)]
            
            patch = patches.Polygon(poly, closed=True, facecolor=color, alpha=0.5, edgecolor='black')
            ax.add_patch(patch)
            
            cx, cy = zone.centroid
            ax.text(cx, cy, f"{zone.zone_id}\nCount: {count:.1f}\nRisk: {risk}", color='white', 
                    ha='center', va='center', fontsize=8, weight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
                    
        ax.set_title("End-to-End Venue Graph Model Prediction")
        ax.set_xlim(0, venue.width)
        ax.set_ylim(venue.height, 0)
        
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved end-to-end plot to {out_file}")

if __name__ == "__main__":
    # Test script execution
    print("Initializing End-to-End Venue Graph Model...")
    model = EndToEndVenueGraphModel()
    
    img_path = r"C:\Users\USER\Downloads\beyond_surveillance_mvp\beyond_surveillance\ShanghaiTech\part_B\train_data\images\IMG_1.jpg"
    venue_path = "configs/venues/dronecrowd_analysis_grid.json"
    
    print(f"Running prediction and plotting for {img_path}...")
    img = cv2.imread(img_path)
    if img is not None:
        model.predict_and_plot(img, venue_path)
    else:
        print("Image not found, skipping plot generation.")
