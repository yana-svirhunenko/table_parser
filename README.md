**Table Detection Demo**

A lightweight web application for detecting table bounding boxes in single-page PDFs using the Microsoft Table Transformer model.


Features

- Detects and visualizes table bounding boxes on uploaded PDFs.
- Saves processed images (with bounding boxes) and JSON metadata (table count, coordinates, and filenames) to the outputs/ folder.
- Displays a message if no tables are detected.


Setup

For easy installation clone this repository. Then create and activate a virtual environment inside the repository folder (recommended):
python -m venv venv  
source venv/bin/activate      # Linux/Mac  
venv\Scripts\activate         # Windows  


Install dependencies using the following command:
pip install -r requirements.txt  


Run the app:

python -m src.app 
Open http://localhost:5001 in your browser. Port can be changed if needed.


Docker

Build and run the container:
docker-compose build  
docker-compose up  
Access the app at http://localhost:5001.


Usage

- Upload a single-page PDF via the web interface.
- Wait for processing (may take a few seconds).
- View the detected tables (if any) or a "no tables found" message.
- Check the outputs/ folder for results.


Limitations

While the Table Transformer performs well for clearly defined, low-resolution tables, I encountered challenges in achieving 
accurate detection with the provided examples. Further configuration adjustments may be necessary to handle such cases effectively. 
For highly specific table formats, fine-tuning the model could improve results.
