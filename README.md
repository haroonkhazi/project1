# project1
## Tuning with RandomizedSearchCV:
This project includes scripts that found the best parameters using random search cv.
To run the random search scripts, you need to have the training data in a csv file, and in the same
directory as the script. To run random search for K Nearest Neighbours, run:
`$: python3.7 knn_classification.py`
For random search for Random Forest run:
`$: python3.7 rf_classification.py`
This will print out the best parameters. These scripts will take a very long time.

## Running Main Classification Code:
We have already found these best parameters by running these scripts, they are in the text file rf_knn_params.txt
So to classify the testing data with the best params just need to run main_classification.py. 
To run this script, have the testing and training data in a csv file and within the same directory as the script, then:
`$: python3.7 main_classification.py.`
This will create 2 files, testingwlabels_knc.csv and testingwlabels_rfc.csv which coresspond to KNN and Random Forest respectively.
These csv files will have the predicted label as under the 'label' column.
