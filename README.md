# Malicious URL Detection
## Project Description
This project was built around developing a Machine Learning model that
could detect when a specific URL is considered malicious. 
The data contains multiple classifications 
Phishin, Defacement, Malware and Bengin.
For this task I focused on building a way to organize the data in such a way
each keyword could be attached to specific values using a Tokenization methodology
similar to ones found in modern large language models.
The final result were two separate algorithms.
Model 1 is a Gradient Boosted Decision Forest that uses the XGBoost library
Model 2 is a Long Short Term Memory Neural Network model that is designed for sequential data.