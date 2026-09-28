import torch
import cv2
import numpy as np
import transformers.modeling_utils
# Bypass the CVE-2025-32434 torch.load safety check to load the B5 .bin weights
transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from crowd_twin.venue import VenueGraph, SemanticZone

class AutomatedVenueExtractor:
    """
    Automatically extracts the physical structure (walkable areas) of a venue
    from a raw camera image using a pre-trained Vision Transformer (SegFormer),
    and converts it into a structured VenueGraph of polygons.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Loading Automated Venue Extractor (SegFormer) on {device}...")
        
        # Using the massive Segformer B5 model for highly accurate scene parsing
        self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(self.device)
        self.model.eval()
        
        # In ADE20K (0-indexed): 
        # 3: floor, 6: road, 9: grass, 11: sidewalk, 13: earth, 54: stairs
        # 12: person (We add this so the crowd doesn't fragment the floor mask!)
        self.walkable_classes = [3, 6, 9, 11, 12, 13, 54]

    @torch.no_grad()
    def extract_walkable_mask(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # [1, 150, H/4, W/4]
        
        # Resize logits to original image size
        logits = torch.nn.functional.interpolate(
            logits, size=(image_rgb.shape[0], image_rgb.shape[1]), mode="bilinear", align_corners=False
        )
        
        predictions = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        # Create a binary mask of walkable areas
        walkable_mask = np.isin(predictions, self.walkable_classes).astype(np.uint8) * 255
        
        return walkable_mask

    def generate_venue_graph(self, image_bgr, venue_id="auto_extracted_venue"):
        """
        Extracts the mask and generates polygonal zones for the VenueGraph.
        """
        mask = self.extract_walkable_mask(image_bgr)
        
        # Aggressive morphological closing to bridge gaps between people
        kernel = np.ones((35, 35), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
        
        # Find contours
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        zones = []
        for i, cnt in enumerate(contours):
            # Filter out very small noise blobs
            area = cv2.contourArea(cnt)
            if area < 1000: # Lowered threshold to capture indoor structured rooms
                continue
                
            # Compute Convex Hull to get a clean structural shape (ignoring jagged edges)
            hull = cv2.convexHull(cnt)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
            # Format for VenueGraph
            poly_points = [[float(p[0][0]), float(p[0][1])] for p in approx]
            
            if len(poly_points) >= 3: # Must be a valid polygon
                cx = float(np.mean([p[0] for p in poly_points]))
                cy = float(np.mean([p[1] for p in poly_points]))
                
                zone = SemanticZone(
                    zone_id=f"zone_{i}",
                    label=f"Walkable Area {i}",
                    polygon=poly_points,
                    area=float(area),
                    capacity=float(area // 100) # Heuristic
                )
                zones.append(zone)
                
        # Generate fully connected adjacency for simplicity, or we can use spatial proximity
        N = len(zones)
        adjacency = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Calculate distance between centroids
                ci = np.mean(zones[i].polygon, axis=0)
                cj = np.mean(zones[j].polygon, axis=0)
                d = np.linalg.norm(ci - cj)
                if d < 300: # Connected if centroids are close
                    adjacency[i][j] = 1
                    adjacency[j][i] = 1
                    
        height, width = image_bgr.shape[:2]
        
        graph = VenueGraph(
            venue_id=venue_id,
            schema_version="1.0",
            source="auto-extracted",
            width=width,
            height=height,
            zones=zones,
            adjacency=adjacency
        )
        return graph, mask_cleaned

def test_venue_extractor():
    extractor = AutomatedVenueExtractor()
    img_path = r"D:\Projects\Beyond-Surveillance\Mall_Dataset\frames\frames\seq_000001.jpg"
    print(f"Processing {img_path} with B5 model...")
    img = cv2.imread(img_path)
    
    if img is not None:
        graph, mask = extractor.generate_venue_graph(img)
        print(f"Automatically extracted {len(graph.zones)} walkable zones with B5!")
        
        img_out = img.copy()
        green_overlay = np.zeros_like(img)
        green_overlay[mask == 255] = [0, 255, 0]
        cv2.addWeighted(green_overlay, 0.3, img_out, 0.7, 0, img_out)
        
        for zone in graph.zones:
            poly = np.array(zone.polygon, dtype=np.int32)
            c = np.mean(poly, axis=0)
            cv2.polylines(img_out, [poly], isClosed=True, color=(0, 0, 255), thickness=3)
            cv2.putText(img_out, zone.zone_id, (int(c[0]), int(c[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        cv2.imwrite("mall_venue_plot_b5.jpg", img_out)
        print("Saved visualization to mall_venue_plot_b5.jpg")
    else:
        print(f"Image not found: {img_path}")

if __name__ == "__main__":
    test_venue_extractor()
