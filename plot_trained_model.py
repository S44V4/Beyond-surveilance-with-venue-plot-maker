import torch
import cv2
from venue_graph_model import EndToEndVenueGraphModel

def run_trained_inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading trained model on {device}...")
    
    # Initialize model
    model = EndToEndVenueGraphModel(device=device).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load("shanghaitech_venue_model.pt", map_location=device))
    model.eval()
    
    img_path = r"C:\Users\USER\Downloads\beyond_surveillance_mvp\beyond_surveillance\ShanghaiTech\part_B\train_data\images\IMG_1.jpg"
    venue_path = "configs/venues/dronecrowd_analysis_grid.json"
    
    print(f"Running prediction and plotting for {img_path}...")
    img = cv2.imread(img_path)
    if img is not None:
        model.predict_and_plot(img, venue_path, out_file="trained_venue_plot.jpg")
    else:
        print("Image not found!")

if __name__ == "__main__":
    run_trained_inference()
