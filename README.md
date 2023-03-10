# Covid19 Malaysia Cases Prediction


 
 



![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


[Description](https://github.com/deenazati08/Covid19-Cases-Prediction#description) // [Result](https://github.com/deenazati08/Covid19-Cases-Prediction#result) // [Credits](https://github.com/deenazati08/Covid19-Cases-Prediction#credits)


## Description

1. Data Loading:

        • Load the dataset to python by import os and pandas.

2. Data Inspection:
    
        • Check the data if there any duplicates and any other anomalities.

3. Data Cleaning:

        • This dataset has some NaN values, so interpolation method were used to replace the NaN values.
        
4. Data Preprocessing:
    
        • First, Normalization were applied to the train data using MinMaxScaler.
    
        • Then, develop the model using Sequential API
<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/207805171-5538544d-22b1-4a9a-b12e-51eb0ff42120.png">
</p> 

        • Callbacks function were use. (Earlystopping and TensorBoard)

        • Predict the data

Last but not least, save all the model used 


## Result
MAE and MAPE result :
<img src="https://user-images.githubusercontent.com/120104404/207801689-c1fdd01c-df4a-46a1-b4be-45b7829af269.jpg">

Predicted and Actual Covid Cases Graph :
<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/207809011-047818c1-09e2-454b-a7fc-89b90fd0f5e0.jpg">
</p>

Loss and MAPE Graph from TensorBoard :
<p align="center">        
<img src="https://user-images.githubusercontent.com/120104404/207801869-a748f43a-2f3c-4550-aab3-e2fac52624f8.jpg">
<img src="https://user-images.githubusercontent.com/120104404/207801895-b9002c4c-b63e-4adf-9e95-a1063a678beb.jpg">
</p>

## Credits

- https://github.com/MoH-Malaysia/covid19-public
