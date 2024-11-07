🌱 Plant Leaf Disease Detection

A machine learning project to detect diseases in plant leaves using ResNet-99 and deploy a prediction model as a web service. This project is designed to help users upload an image of a plant leaf, get a prediction of the disease, and receive additional information on how to treat it.
📋 Project Overview

The saved model's h5 file was too large to be included in the repository. So if you're trying out the project, just re-train the model and save the model to plant-disease-api/model/EfficientNetB3-Plant Disease-98.40.h5

This project leverages deep learning and computer vision to classify plant leaf diseases from images. The primary goal is to provide a tool for farmers, researchers, and plant enthusiasts to easily identify diseases from images and learn how to manage them.
Features

    Image Classification: Trained ResNet-99 model to classify leaf diseases.
    Backend API: Model hosted as a REST API using FastAPI.
    Frontend Webpage: A ReactJS interface where users can upload leaf images and receive a disease diagnosis along with additional information.

📁 Dataset

The dataset for training and testing the model is sourced from Kaggle, containing labeled images of healthy and diseased plant leaves across various plant species.
🧪 Model Architecture

The model is built using ResNet-99, a deep convolutional neural network architecture that is highly effective for image classification tasks. The model was trained in Jupyter Labs using Python.
🖥️ Project Structure

├── frontend/             # ReactJS frontend
├── backend/              # FastAPI backend server
│   ├── app.py            # API endpoints for prediction
│   └── model/            # Saved ResNet-99 model
├── data/                 # Dataset (sourced from Kaggle)
└── notebooks/            # Jupyter notebooks for model training

🚀 Getting Started
Prerequisites

    Python 3.7+
    Node.js & npm
    Jupyter Notebook
    FastAPI and uvicorn
    ReactJS

Installation

    Clone the repository

git clone https://github.com/your-username/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection

Set up Backend

    Navigate to the backend directory:

cd backend

Install dependencies:

pip install -r requirements.txt

Start the FastAPI server:

    uvicorn app:app --reload

Set up Frontend

    Navigate to the frontend directory:

cd frontend

Install dependencies:

npm install

Start the React app:

        npm start

📊 Usage

    Upload an Image: Use the ReactJS webpage to upload a plant leaf image.
    Receive Prediction: The image is sent to the backend, where the ResNet-99 model processes it and predicts the disease.
    View Details: The result shows the disease type, description, treatment options, and relevant links for more information.

🧩 Model Training

The model was trained on the Kaggle dataset using Jupyter Notebook. Training involves preprocessing images, data augmentation, and tuning the ResNet-99 model.

    Model Architecture: ResNet-99 with adjustments for multi-class plant disease classification.
    Training Environment: Python in Jupyter Labs.

Refer to the notebooks/ directory for the training notebook.
🛠 Technologies Used

    FastAPI - Backend API server
    ReactJS - Frontend framework
    ResNet-99 - Deep learning model
    Kaggle - Data source
    Jupyter Labs - Model training environment

🌐 Links and References

    Kaggle Dataset
    FastAPI Documentation
    ReactJS Documentation

🤝 Contributing

Feel free to open issues or submit pull requests for improvements!
📄 License

This project is licensed under the MIT License. See the LICENSE file for details.