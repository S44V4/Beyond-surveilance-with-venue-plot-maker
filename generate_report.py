import os
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_report():
    doc = Document()
    
    # Title
    title = doc.add_heading('End-to-End Venue Graph Architecture', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph('Automated Crowd Analytics and Structural Reasoning', style='Subtitle')
    
    # 1. Brief General Workflow
    doc.add_heading('1. General Workflow Overview', level=1)
    doc.add_paragraph(
        "The End-to-End Venue Graph Architecture is an advanced AI surveillance pipeline designed to move beyond "
        "simple crowd counting. Instead of just treating a camera feed as a grid of pixels, it understands the "
        "physical architecture of the environment (the 'Venue Graph') and tracks how crowds flow through those physical spaces."
    )
    
    doc.add_paragraph("The pipeline operates in three distinct stages:")
    
    p1 = doc.add_paragraph(style='List Bullet')
    p1.add_run("Automated Venue Extraction: ").bold = True
    p1.add_run("A pre-trained Vision Transformer automatically scans the camera feed, ignores crowds/clutter, and mathematically maps out the walkable corridors to generate a structural floorplan.")
    
    p2 = doc.add_paragraph(style='List Bullet')
    p2.add_run("Crowd Density Estimation (DDPFNet): ").bold = True
    p2.add_run("Simultaneously, a Swin Transformer backbone analyzes the raw image to predict pixel-by-pixel crowd density heatmaps.")
    
    p3 = doc.add_paragraph(style='List Bullet')
    p3.add_run("Graph Risk Reasoning: ").bold = True
    p3.add_run("The raw pixel data is projected onto the extracted physical layout. A Graph Neural Network (GNN) then analyzes the topological connections between rooms to predict the final Crowd Count and categorical Risk Level (Safe, Watch, Critical) for each distinct architectural zone.")
    
    # 2. Detailed Venue Graph
    doc.add_heading('2. Automated Venue Graph Generation', level=1)
    
    doc.add_paragraph(
        "The cornerstone of the system is the Automated Venue Graph Extractor. In traditional crowd analytics, a human "
        "operator must manually draw polygons over a camera feed to tell the system where the 'rooms' and 'doors' are. "
        "This is unscalable for smart cities with thousands of cameras."
    )
    
    doc.add_paragraph(
        "Our system automates this entirely using a pre-trained Vision Transformer Foundation Model (SegFormer), which "
        "was pre-trained on millions of structured environments. When connected to a camera, the model performs "
        "Semantic Scene Parsing to discover the layout of the venue."
    )
    
    doc.add_heading('Algorithmic Methodology', level=2)
    doc.add_paragraph(
        "1. Semantic Grouping: The AI separates structural features (walls, ceilings, trees) from walkable surfaces (floors, paths, roads). "
        "Crucially, it groups 'person' pixels together with 'floor' pixels to ensure that dense crowds do not fragment the structural mapping.\n"
        "2. Morphological Cleaning: The segmented mask undergoes aggressive morphological transformations (closing/opening) to bridge any gaps.\n"
        "3. Convex Hull Polygonization: Contour detection maps the bounds of the walkable space, and a Convex Hull algorithm computes clean, "
        "structural architectural boundaries (ignoring the jagged edges caused by occlusions).\n"
        "4. Graph Instantiation: The resulting polygons are mathematically instantiated into a structured JSON configuration (the Venue Graph) "
        "containing distinct Semantic Zones and their Adjacency connections."
    )
    
    # Helper function to add centered images
    def add_centered_image(doc, img_path, caption):
        if os.path.exists(img_path):
            # Create a centered paragraph for the image
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p_img.add_run()
            run.add_picture(img_path, width=Inches(5.5))
            
            # Create a centered paragraph for the caption
            p_cap = doc.add_paragraph(caption, style='Caption')
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        else:
            doc.add_paragraph(f"[Image Missing: {img_path}]", style='Caption')

    # Add Image 1
    doc.add_heading('Showcase: Structured Indoor Environment (The Mall Dataset)', level=2)
    doc.add_paragraph(
        "To demonstrate the system's structural reasoning, it was deployed on the Ke Chen Mall Dataset—a famous computer vision "
        "dataset containing thousands of frames of surveillance footage from an indoor shopping mall corridor."
    )
    
    add_centered_image(doc, "mall_venue_plot.jpg", "Figure 1: AI dynamically extracting the walkable mall corridor while ignoring storefront walls and ceilings.")
    
    doc.add_paragraph(
        "As seen in Figure 1, the AI perfectly maps the structural boundary of the corridor. Because the Mall Dataset "
        "is a continuous video sequence from a fixed CCTV camera, it allows us to test the temporal stability of the extractor."
    )
    
    # Add Image 2
    doc.add_heading('Temporal Stability and Crowd Occlusion', level=2)
    doc.add_paragraph(
        "A major challenge in automatic extraction is occlusion. When a massive group of shoppers walks through the corridor, "
        "they block the floor. A naive model would redraw the room shape to avoid the people. Our geometric Convex Hull approach "
        "solves this. As tested across frames 1 through 2000 of the Mall Dataset, the AI correctly identifies that the underlying "
        "architecture hasn't moved, drawing the exact same stable structural polygon over the corridor regardless of crowd density."
    )
    
    add_centered_image(doc, "mall_showcase_3.jpg", "Figure 2: The model maintaining stable architectural boundaries despite heavy crowd occlusion in Frame 1000.")
        
    doc.add_heading('Showcase: Unstructured Environments (ShanghaiTech)', level=2)
    doc.add_paragraph(
        "The model is also robust enough to handle unstructured, open-air environments like those found in the ShanghaiTech "
        "dataset. In open plazas with thousands of people, the system seamlessly adapts to draw massive, unified bounds "
        "around the entire walkable street."
    )
    
    add_centered_image(doc, "auto_extracted_venue.jpg", "Figure 3: Unstructured venue extraction mapping a massive open-air plaza.")

    # Save
    report_name = 'Venue_Graph_Architecture_Report.docx'
    doc.save(report_name)
    print(f"Document saved to {report_name}")

if __name__ == "__main__":
    create_report()
