
To Run Convolutional Neural Network classification:
    To run the cnn_classification.py, have the training data in a csv file in the same directory. Run:
    $: python3.7 cnn_classification.py
    This will run a tuning grid, GridSearchCV which is an exhaustive search over the possible params for the CNN.
    It will then print the best params and the score of the best parameters.

    To run the model on the best params that we found, have both the testing and training data in the same directory.
    Run:
    $: python3.7 cnn_best_params_classification.py
    This will fit the model to the data in the training data set, and test it as well over 6 epochs. It will get around 98-99% accuracy.
    This will also create a new csv file called testingwlabels_cnn.csv which has appends a column, 'labels', to the testing data.


To Run Naive Bayes:
    To run this script, have the testing and training data in a csv file and within the same directory as the script, then:
    $: python3.7 Naive_Bayes.py
    This will print out an accuracy score on the training data after fitting the data.

To Run svm.py
    To run this script, have the testing and training data in a csv file and within the same directory as the script, then:
    $: python3.7 svm.py
    This will print out an accuracy score on the training data after fitting the data.


To Run Random Forest and K NEarest Neighbours:
    Tuning with RandomizedSearchCV:
        This project includes scripts that found the best parameters using random search cv.
        To run the random search scripts, you need to have the training data in a csv file, and in the same
        directory as the script. To run random search for K Nearest Neighbours, run:
        $: python3.7 knn_classification.py
        For random search for Random Forest run:
        $: python3.7 rf_classification.py
        This will print out the best parameters. These scripts will take a very long time.

    Running knn_rf_classification.py Code:
        We have already found these best parameters by running these scripts, they are in the text file rf_knn_params.txt
        So to classify the testing data with the best params just need to run knn_rf_classification.py.
        To run this script, have the testing and training data in a csv file and within the same directory as the script, then:
        $: python3.7 knn_rf_classification.py.
        This will create 2 files, testingwlabels_knc.csv and testingwlabels_rfc.csv which coresspond to KNN and Random Forest respectively.
        These csv files will have the predicted label as under the 'label' column.
        It will also print out the accuracy score for the Random Forest classification on the training data and then print out an accuracy score for KNN classification.py for the training data as well.
        Finally after predicting the testing.csv data, it will print the accuracy between the two classification methods on the testing data.


To Run decision Tree classification:
    To run the tuning grid we used for the decision tree, have the training data in a csv file in the same directory. Run:
    $: python3.7 decision_tree_classification.py
    it will then print the best params and the score of the best parameters.

    To run the modle on the best params we found, have the training data in the same directory.
    Run:
    $: python3.7 dt_best_params_classification.py
    it will print the score of the best params.
