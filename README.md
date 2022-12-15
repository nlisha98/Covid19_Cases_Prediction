## Badges
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) 
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) 
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white) 
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

# Covid 19 Cases Prediction
 Deep Learning project on how to predict new cases of Covid-19.
 
## Project Description
The purpose of this project is to create a Deep Learning model by using LSTM neural network to predict new cases of Covid-19 in Malaysia using the past 30 days of number of cases.

### Project Details 
(A). Data Loading <br>

<ul>
  <li>Load the dataset into this project using pandas.</li>
</ul>

(B). Data Inspection <br>

<ul>
  <li>Check the data type of 'cases_new' column - object datatype</li>
  <li>Check if the dataset has duplicated data - found 0 duplicates data</li>
  <li>Check if the dataset has any missing values - 12 missing values found</li>
</ul>

(C). Data Cleaning <br>

<ul>
  <li>Change the data type of 'cases_new' column to int64</li>
  <li>Fill in missing values - by using Interpolation</li>
</ul>

(D). Features Selection <br>

<ul>
  <li>No features to select</li>
</ul>

(E). Data Pre-Processing <br>

<ul>
  <li>Expand the dimension of 'cases_new' column using numpy</li>
  <li>Normalization by using Min-Max Scaling</li>
  <li>Fit and Transform the data</li>
  <li>Define the x train and y train</li>
  <li>Train Test Split the data</li>
</ul>

(F). Model Development <br>

<ul>
  <li>Create Sequential Model</li>
  <li>Add LSTM layers</li>
  <li>Add Dropout layers</li>
  <li>Add Dense layers</li>
  <li>Model Summary</li>
</ul>

(G). Model Compilation <br>

<ul>
  <li>Compile the model</li>
    <ul>
      <li> Optimizer - adam </li>
      <li> Loss - 'mse'</li>
      <li> Metrics - ['mse','mape']</li>
    </ul>
</ul>

(H). Callbacks - Early Stopping and TensorBoard<br>

<ul>
  <li>Tensorboard logs after every batch of training to monitor metrics</li>
  <li>Save model to disk</li>
</ul>

(I). Model Training <br>

<ul>
  <li>Train the model for 100 epochs and get the best val_loss</li>
</ul>

(J). Model Evaluation <br>

<ul>
  <li>Plot the training and validation loss</li>
  <li>Load and Inspect testing dataset</li>
  <li>Concatenation the train and test data</li>
  <li>Get the model prediction</li>
  <li>Visualize the predicted new cases and actual new cases</li>
  <li>Evaluate the performance by using MAPE and MAE</li>
</ul>

(K). Model Saving

<ul>
  <li>Save the model</li>
    <ul>
      <li>Min-Max Scaler</li>
      <li>Save Model</li>
    </ul>
</ul>



## Results
### Model Architecture

<p align="center">
   <img
    src="Model/model_architecture.png"
    alt="Model Architecture"
    title="Model Architecture"
    width=300>
 </p>
  
  
### Model Performance

<p align="center">
  <img
  src="Model/testing_evaluation_metrics.PNG"
  alt="Model Performance"
  title="Model Performance"
  width = 300>
</p>
    
### Predicted and Actual Graph

<p align="center">
  <img
  src="Model/graph_visualization.PNG"
  alt="Graph Visualization"
  title="Graph Visualization"
  class = "center">
</p>
 
## Acknowledgement
Special thanks to **(GitHub - MoH-Malaysia/covid19-public: Official data on the COVID-19
epidemic in Malaysia. Powered by CPRC, CPRC Hospital System,
MKAK, and MySejahtera)** for the dataset used for this project.

Link: https://github.com/MoH-Malaysia/covid19-public
