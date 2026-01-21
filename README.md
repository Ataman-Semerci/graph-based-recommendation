# ⚠️ Important Notice About Local File Paths

This project was developed and tested in a local Windows environment.

Some scripts and notebooks include absolute local file paths such as:
C:/Users/.../Desktop/...

If you clone this repository, you may need to update file paths
according to your own directory structure before running the code.

This is expected behavior and does not indicate an installation error.

---

# Graph-Based Outfit Recommendation System

This project was developed within the scope of the
TÜBİTAK 2209-A Undergraduate Research Projects Support Program.

The goal of this study is to recommend clothing outfits by modeling
item–outfit relationships as a graph structure and combining them
with visual similarity extracted from product images.

---

## Project Overview

The system operates in two main stages.

First, visual features are extracted from clothing images using ResNet-50.
Each clothing item is represented as a high-dimensional embedding vector.

Second, items and outfits are modeled as nodes in a graph structure.
Edges represent compatibility relationships between items and outfits.
Recommendations are generated using cosine similarity and graph
neighborhood traversal.

A graphical user interface is provided to demonstrate the system
in an interactive manner.

---

## Project Structure

graph-based-outfit-recommendation/

- notebooks  
  - graph-based-recommendation-system.ipynb  

- OutfitGUI  
  - app.py  
  - run_app.bat  
  - data  
    - item_features.npy  
    - outfit_graph.gpickle  
  - assets  
    - tubitak_logo.png  

- test_images  

---

## Running the GUI

Requirements:
Python 3.9 – 3.11 recommended  
Windows OS (tested)

Install dependencies using:

pip install -r requirements.txt

Run the application with:

streamlit run app.py

or simply double-click:

run_app.bat

---

## Dataset Information

Due to GitHub file size limitations, the original image dataset
is not included in this repository.

The repository includes precomputed item features
(item_features.npy) and a preconstructed outfit graph
(outfit_graph.gpickle).

These files allow the system to be demonstrated without re-running
the full feature extraction pipeline.

If you would like to run the full pipeline from scratch, 
please open an Issue in this repository to request access to the original dataset.

---

## Notebook Description

The notebook under the notebooks directory demonstrates
image preprocessing, feature extraction, graph construction
and similarity-based recommendation logic.

Some cells include local file paths that must be updated
before execution on another machine.

---

## Academic Context

This project was developed as part of an undergraduate research study
supported by TÜBİTAK 2209-A.

---

## Technologies Used

Python  
PyTorch  
Torchvision  
NetworkX  
NumPy  
Pandas  
Scikit-learn  
Streamlit  

---

## Author

Ataman Semerci  
Electrical & Electronics Engineering  
Izmir Democracy University
## Contact

For academic or research-related inquiries, please contact me via GitHub Issues.

