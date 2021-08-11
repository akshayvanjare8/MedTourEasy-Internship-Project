# Data-Analyst-Project-Proposal-at-MedTourEasy

![15254_full](https://user-images.githubusercontent.com/72245329/129059539-db2a9764-c822-434b-984f-587bb2fee723.png)



# Project Description 
Forecasting blood supply is a serious and recurrent problem for blood collection managers.In this Project, you will work with data collected from the donor database of Blood TransfusionService Center.  The dataset, obtained from the Machine Learning Repository, consists of arandom sample of 748 donors. Your task will be to predict if a blood donor will donate within a giventime window. You will look at the full model-building process: from inspecting the dataset to usingthe tpot library to automate your Machine Learning pipeline.To complete this Project, you need to know some Python, pandas, and logistic regression.

# Process
We will work closely with you to build and fulfill the needs of this project by the end of your internship.We will do this by establishing clear goals and a comprehensive solution based on projectrequirements.
Our process to achieve this is as follows:

# Task1:Instructions
Inspect the file that contains the dataset.
•Print out the first 5 lines from datasets/transfusion.data using the head shellcommand.Make sure to first read the narrative for each task in the notebook on the right before reading themore detailed instructions here. To complete this Project, you need to know some Python, pandas,and logistic regression. We recommend one is familiar with the content.To run a shell command in a notebook, you prefix it with !, e.g. !ls will list directory contents.

# Task2:Instructions
Load the dataset.
•Import the pandas library.
•Load the transfusion.data file from datasets/transfusion.data and assign it tothe transfusion variable.
•Display the first rows of the DataFrame with the head() method to verify the file was loadedcorrectly.
If you print the first few rows of data, you should see a table with only 5 columns.

# Task3:Instructions
Inspect the DataFrame's structure.
•Print a concise summary of the transfusion DataFrame with the info() method.DataFrame's info() method prints some useful information about a DataFrame:
•index type
•column types
•non-null values
•memory usageincluding the index dtype and column dtypes, non-null values and memoryusage.

# Task4:Instructions
Rename a column.
•Rename whether he/she donated blood in March 2007 to target for brevity.
•Print the first 2 rows of the DataFrame with the head() method to verify the change was donecorrectly.By setting the inplace parameter of the rename() method to True, the transfusion DataFrameis changed in-place, i.e., the transfusion variable will now point to the updated DataFrame asyou'll verify by printing the first 2 rows.

# Task5:Instructions
Print target incidence.•Use value_counts() method on transfusion.target column to print target incidenceproportions, setting normalize=True and rounding the output to 3 decimal places.By default, value_counts() method returns counts of unique values. Bysetting normalize=True, the value_counts() will return the relative frequencies of the uniquevalues instead.

# Task6:Instructions
Split the transfusion DataFrame into train and test datasets.
•Import train_test_split from sklearn.model_selection module.
•Split transfusion into X_train, X_test, y_train and y_test datasets, stratifying onthe target column.
•Print the first 2 rows of the X_train DataFrame with the head() method.Writing the code to split the data into the 4 datasets needed would require a lot of work. Instead, youwill use the train_test_split() method in the scikit-learn library.

# Task7:Instructions
Use the TPOT library to find the best machine learning pipeline.
•Import TPOTClassifier from tpot and roc_auc_score from sklearn.metrics.
•Create an instance of TPOTClassifier and assign it to tpot variable.
•Print tpot_auc_score, rounding it to 4 decimal places.
•Print idx and transform in the for-loop to display the pipeline steps.You will adapt the classification example from the TPOT's documentation. In particular, you willspecify scoring='roc_auc' because this is the metric that you want to optimize for andadd random_state=42 for reproducibility. You'll also use TPOT lightconfiguration with only fastmodels and preprocessors.The nice thing about TPOT is that it has the same API as scikit-learn, i.e., you first instantiate amodel and then you train it, using the fit method.Data pre-processing affects the model's performance, and tpot's fitted_pipeline_ attribute willallow you to see what pre-processing (if any) was done in the best pipeline.

# Task8:InstructionsCheck the variance.
•Print X_train's variance using var() method and round it to 3 decimal places.pandas.DataFrame.var() method returns column-wise variance of a DataFrame, which makescomparing the variance across the features in X_train simple and straightforward.

# Task9:Instructions
Correct for high variance.
•Copy X_train and X_test into X_train_normed and X_test_normed respectively.•Assign the column name (a string) that has the highest varianceto col_to_normalize variable.•For X_train and X_test DataFrames:
•Log normalize col_to_normalize to add it to the DataFrame.•Drop col_to_normalize.
•Print X_train_normed variance using var() method and round it to 3 decimal places.X_train and X_test must have the same structure. To keep your code "DRY" (Don't RepeatYourself), you are using a for-loop to apply the same set of transformations to each of theDataFrames.Normally, you'll do pre-processing before you split the data (it could be one of the steps in machinelearning pipeline). Here, you are testing various ideas with the goal to improve model performance,and therefore this approach is fine.

# Task 10: Instructions
Train the logistic regression model.
•Import linear_model from sklearn.
•Create an instance of linear_model.LogisticRegression and assign itto logreg variable.
•Train logreg model using the fit() method.
•Print logreg_auc_score.The scikit-learn library has a consistent API when it comes to fitting a model:1.Create an instance of a model you want to train.2.Train it on your train datasets using the fit method.You may recognise this pattern from when you trained TPOT model. This is the beauty ofthe scikit-learn library: you can quickly try out different models with only a few code changes.

# Task 11: Instructions
Sort your models based on their AUC score from highest to lowest.
•Import itemgetter from operator module.
•Sort the list of (model_name, model_score) pairs from highest to lowestusing reverse=True parameter.Congratulations, you've made it to the end!Good luck and keep on learning!If you are interested in learned what makes linear models so powerful and widely used, Statistical Modeling in R is a great resource! The coding is done in R, but it's the theoretical concepts that willhelp you to interpret the models you are building.
