# CharityNeuralNetwork

### CHALLENGE

this challenge set out to answer the following questions, see the APPENDIX for details of the challenge.

How many neurons and layers did you select for your neural network model? Why?
- Initialy choose 62 neurons in one hidden layer, where 62 was twice the number of input neurons as per recommended best practice.
- Choose one layer to intially save computation resource, reduce time taken to compute results and reduce the liklihood of overfitting. 

Were you able to achieve the target model performance? What steps did you take to try and increase model performance?
- Initial performance metric for accurancy was 56%, which then did not budge through all the various tactics trying to improve performance
  - Increased number of hidden layer neurons
  - Increased the number of hidden layers
  - Tried the variuos hidden layer activation functions
  - Increased the number of epochs
  - Started to check the data for possible outliers
  
If you were to implement a different model to solve this classification problem, which would you choose? Why?
- I would recomend trying SVMs next, which can build adequate models with linear or nonlinear data. Due to SVMs’ ability to create multidimensional borders, SVMs lose their interpretability and behave more like the black box machine learning models, such as basic neural networks and deep learning models. SVMs perform one task and one task very well — they classify and create regression using two groups.

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
### Give Your Model a Synaptic Boost
There are a few means of optimizing a neural network:
- Check out your input dataset
  - Even if we standardize and scale our numerical variables, too many outliers in a single variable can lead to performance issues
  - look for outliers that can help identify if a particular numerical variable is causing confusion in a model. Try leaving out a noisy variable from the rest of the training features and see if the model performs better.
- Add more neurons to a hidden layer
  - Adding neurons to a hidden layer has diminishing returns—more neurons means more data as well as a risk to overfitting the model.
- Add additional hidden layers
  - Start with one, then increase the number of layers one at a time
  - Changing the structure of the model by adding additional hidden layers allows neurons to train on activated input values, instead of looking at new training data
- Use a different activation function for the hidden layers
  - Options are: 
  - sigmoid function values are normalized to a probability between 0 and 1, which is ideal for binary classification.
    - This function is identified by a characteristic S curve. It transforms the output to a range between 0 and 1.
  - tanh function can be used for classification or regression, and it expands the range between -1 and 1.
    - This function is identified by a characteristic S curve; however, it transforms the output to a range between -1 and 1.
  - ReLU function is ideal for looking at positive nonlinear input data for classification or regression.
    - This function returns a value from 0 to infinity, so any negative input through the activation function is 0. It is the most used activation function in neural networks due to its simplifying output, but might not be appropriate for simpler models.
  - Leaky ReLU function is a good alternative for nonlinear input data with many negative inputs.
    - This function is an alternative to another activation function, whereby negative input values will return very small negative values.
- Add additional epochs to the training regimen
  - As the number of epochs increases, so does the amount of information provided to each neuron. Be wary about overfitting.


**IMPORTANT**
You will not be penalized if your model does not achieve target performance, as long as you demonstrate an attempt at model optimization within your notebook.

Create a new README.txt file within the same folder as your AlphabetSoupChallenge.ipynb notebook. Include a 5–10 sentence writeup in your README that addresses the following questions:
- How many neurons and layers did you select for your neural network model? Why?
- Were you able to achieve the target model performance? What steps did you take to try and increase model performance?
- If you were to implement a different model to solve this classification problem, which would you choose? Why?
