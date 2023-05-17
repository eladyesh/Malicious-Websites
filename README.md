# Malicious Website Detection with Neural Network

This project is a neural network implementation in Python to detect malicious websites. The program extracts data from a URL, processes it, and uses TensorFlow and datasets to predict if the website is malicious or not.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Pandas
- Scikit-learn
- Requests
- BeautifulSoup4

## Usage

1. Clone the repository: `git clone https://github.com/eladyesh/Malicious-Websites.git`
2. Install the required packages using pip: `pip install -r requirements.txt`

#### A single URL
run the program like the following:
`python network.py [-h] url` where url is the parameter

#### Multiple URL's
run the program like the following:
`python multiple_domains.py [-h] domains.txt` where domains.txt is the parameter
an output file with the analysis of each url will be presented.

## How it Works

The program uses a neural network model built with TensorFlow to predict if a website is malicious or not. The model is trained on a dataset of known malicious and benign websites.

The program extracts data from the URL, such as the HTML content and various website features, and processes it into a format that can be used by the neural network. The model then makes a prediction based on the input data.

## Dataset

The dataset used to train the neural network is sourced from the University of New Mexico and contains over 5000 website samples labeled as either malicious or benign. The dataset can be found in the `dataset` directory.

## Credits

This project was created by eladyesh as part of a cybersecurity course.
