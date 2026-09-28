import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def render_venue_minimap(json_path, output_path="venue_minimap.jpg"):
    # Load the venue graph JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    zones = data.get('zones', [])
    
    # Setup plot with a clean, dark "radar" style
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Colors for different risk levels (simulated for the minimap)
    # Let's say Zone 4 (center) is HIGH risk, others are SAFE
    risk_colors = {
        'SAFE': '#2ecc71',    # Green
        'WATCH': '#f1c40f',   # Yellow
        'HIGH': '#e74c3c'     # Red
    }
    
    for i, zone in enumerate(zones):
        poly_points = np.array(zone['polygon'])
        
        # Simulate some predictions for the showcase
        risk = 'SAFE'
        if i == 4:
            risk = 'HIGH'
        elif i in [3, 5]:
            risk = 'WATCH'
            
        color = risk_colors[risk]
        
        # Draw the zone polygon
        polygon = Polygon(poly_points, closed=True, 
                         facecolor=color, alpha=0.3, 
                         edgecolor=color, linewidth=3)
        ax.add_patch(polygon)
        
        # Calculate centroid for text
        cx = np.mean(poly_points[:, 0])
        cy = np.mean(poly_points[:, 1])
        
        # Add labels
        ax.text(cx, cy, f"ZONE {i+1}", 
                color='white', fontweight='bold', fontsize=12,
                ha='center', va='center')
        
        ax.text(cx, cy + 20, f"Status: {risk}", 
                color=color, fontsize=10,
                ha='center', va='center')

    # Draw connections (adjacency)
    adjacency = data.get('adjacency', [])
    if adjacency:
        for i in range(len(zones)):
            for j in range(i + 1, len(zones)):
                if adjacency[i][j]:
                    p1 = np.mean(np.array(zones[i]['polygon']), axis=0)
                    p2 = np.mean(np.array(zones[j]['polygon']), axis=0)
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                            color='white', linestyle='--', alpha=0.5, linewidth=1.5)

    # Styling
    ax.set_title("Venue 2D Layout & Live Risk Map", color='white', fontsize=16, pad=20)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Image coordinates start top-left
    
    # Remove messy axes
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(2)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
    plt.close()
    print(f"Saved minimap to {output_path}")

if __name__ == "__main__":
    json_path = "configs/venues/dronecrowd_analysis_grid.json"
    render_venue_minimap(json_path)
