# CharityNeuralNetwork

### CHALLENGE


-
-
-
-
-
-
-
-
-
# APPENDIX

### Module 19 Challenge
Beks is finally ready to put her skills to work to help the foundation predict where to make investments. She’s come a long way since her first day at that boot camp five years ago—and since her first day at Alphabet Soup—and since earlier this week, when she just started learning about neural networks!

Who knows how much farther she will go?

In this challenge, you’ll have to build your own machine learning model that will be able to predict the success of a venture paid by Alphabet soup. Your trained model will be used to determine the future decisions of the company—only those projects likely to be a success will receive any future funding from Alphabet Soup.

### Background
From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received various amounts of funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

Using your knowledge of machine learning and neural network model building, you must create a **binary classifier** that is capable of predicting whether or not an applicant will be successful if funded by Alphabet Soup using the features collected in the provided dataset.

### Objectives
The goals of this challenge are for you to:

- Import, analyze, clean, and preprocess a “real-world” classification dataset.
- Select, design, and train a binary classification model of your choosing.
- Optimize model training and input data to achieve desired model performance.

### Instructions
- Create a new Jupyter Notebook within a new folder on your computer. 
- Name this new notebook file **“AlphabetSoupChallenge.ipynb”** (or something easily identifiable).
- Download the Alphabet Soup Charity dataset (charity_data.csv) and place it in the same directory as your notebook.
- Import and characterize the input data.

**IMPORTANT**
Be sure to identify the following in your dataset:

- What variable(s) are considered the target for your model?
- What variable(s) are considered to be the features for your model?
- What variable(s) are neither and should be removed from the input data?

**Preprocess the data**
Using the methods described in this module, preprocess all numerical and categorical variables, as needed:
- Combine rare categorical values via bucketing.
- Encode categorical variables using one-hot encoding.
- Standardize numerical variables using Scikit-Learn’s StandardScaler class.

**Neural Network design**
Using a TensorFlow neural network design of your choice, create a binary classification model that can predict if an Alphabet Soup funded organization will be successful based on the features in the dataset.
- You may choose to use a **neural network** or **deep learning** model.

**HINT:**
Think about how many inputs there are before determining the number of neurons and layers in your model.

Compile, train, and evaluate your binary classification model. Be sure that your notebook produces the following outputs:
- Final model loss metric
- Final model predictive accuracy
- Do your best to optimize your model training and input data to achieve a **target predictive accuracy higher than 75%**.

Look at Page 19.2.6 for ideas on how to optimize and boost model performance.

**IMPORTANT**
You will not be penalized if your model does not achieve target performance, as long as you demonstrate an attempt at model optimization within your notebook.

Create a new README.txt file within the same folder as your AlphabetSoupChallenge.ipynb notebook. Include a 5–10 sentence writeup in your README that addresses the following questions:
- How many neurons and layers did you select for your neural network model? Why?
- Were you able to achieve the target model performance? What steps did you take to try and increase model performance?
- If you were to implement a different model to solve this classification problem, which would you choose? Why?
