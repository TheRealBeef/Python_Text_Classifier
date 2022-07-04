# Classifier Project

Note: Requires the training data (mlarr_text) folder to be placed into the directory to re-create data.txt - 
	this directory name is hardcoded in global_def.py. Unzip mlarr_text.zip to access the data (from BBC)

## Command Line Arguments

(-s)    Consolidates Data - saves the documents from "mlarr_text" folder (separated by category) into data.txt, and performs
        some basic clean-up of the files as well

(-a)    Analyze data - Performs a Frequency analysis of the data and returns the 20 most common words in each category

(-nbi)  Naive Bayes Iterative - Performs Naive Bayes classification on data.txt, iterating from 5 features and
        increasing the number of features by 10 each time until 90% precision on the test set is achieved

(-nb)   Performs Naive Bayes Classification on data.txt, with the number of features defined by NUM_FEATURES_NB
        in global_def.py

(-nn)   Performs Multi-Layer Perceptron Neural Network classification on data.txt, with the number of features defined
        by NUM_FEATURES_NN in global_def.py

(-go)   Performs an classification of the test document (a CNN Business report) using both the Naive Bayes and Neural Network
        classifiers, pulled from naive_bayes_classifier.pkl/count_vectorizer.pkl for the Naive bayes, and from
        nn_classifier.pkl/count_vectorizer_nn.pkl for the MLP Neural Network

## Example Command Line Statements (in order)
	python3 main.py -s -a		# Takes data from mlarr_text folder, consoldiates to data.txt, then performs frequency analysis
	python3 main.py -nb -nn -go	# Uses the data in data.txt to train Naive Bayes and Neural Network classifiers, 
					  save them to .pkl, then use the trained classifiers in .pkl to classify the test document
	python3 main.py -nbi -go	# Trains a Naive Bayes classifier using as few iterations as possible to reach 90% precision
					  saves it to .pkl, then uses the trained classifiers in.pkl to classify test document. No
					  change is made to the neural network classifier pkl 		

## Files Generated:
naive_bayes_classifier.pkl
nn_classifier.pkl
count_vectorizer.pkl
count_vectorizer_nn.pkl
data.txt

## Non-Standard Libraries Used:
    SCIKIT-LEARN (SKLEARN)
    NLTK
    PICKLE

## If an error is thrown for missing dependencies stopwords and punkt, uncomment the following lines in ext_functions.py
lines 15-17
    # import nltk
    # nltk.download('stopwords')
    # nltk.download('punkt')
