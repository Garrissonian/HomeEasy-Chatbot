# HomeEasy-Chatbot
This repo contains all the necessary files for the specialized chatbot for real estate agency to generate feedback for their employees' performances.

##Note: As I am using cpu for inference due to unavailability of gpu resources it takes some time for giving answer so it is advised to be paitent and api calls should be made one by one. When it gives response for one thing then refresh and give other query to it.  

## Overview

The Sales Performance Chatbot is an AI-powered application designed to provide insights into sales performance data. It leverages natural language processing (NLP) and machine learning techniques to generate responses based on sales data. The application is built using Flask for the backend and utilizes various transformer models for text generation and data processing.

## Architecture

The architecture of the Sales Performance Chatbot consists of the following components:

1. **Backend**:
   - **Flask**: A lightweight WSGI web application framework used to create and manage the API endpoints.
   - **Transformers**: The Hugging Face Transformers library is used for loading and utilizing pre-trained models such as GPT-2 and BERT for text generation and embedding extraction.
   - **FAISS**: A library for efficient similarity search and clustering of dense vectors.
   - **Pandas**: Used for data manipulation and analysis.
   - **Torch**: PyTorch is used as the deep learning framework for model inference.

2. **Frontend**:
   - **HTML/CSS**: Provides a user-friendly interface for interacting with the chatbot.
   - **JavaScript**: Handles form submissions and interacts with the backend API.

## Technologies Used

- **Flask**: For creating RESTful API endpoints.
- **Hugging Face Transformers**: For loading and using pre-trained models like GPT-2 and BERT.
- **FAISS**: For efficient similarity search.
- **Pandas**: For handling and processing sales data.
- **PyTorch**: For running the machine learning models.
- **HTML/CSS/JavaScript**: For the web-based user interface.

## Setup and Run Instructions

### Prerequisites

Make sure you have Python 3.8 or later installed on your system. You also need to have `pip` to install the required packages.

### 1. Clone the Repository
#### Cloning repo
git clone https://github.com/Garrissonian/HomeEasy-Chatbot.git

#### Changing directory to root directory
cd HomeEasy-Chatbot

### 2.Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

### 3.Install Dependencies
pip install -r requirements.txt

### 4.Run the Backend
python api.py


