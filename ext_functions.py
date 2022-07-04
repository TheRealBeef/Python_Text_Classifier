import os
import random
import string
import global_def as glb
import pickle

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from nltk.corpus import stopwords
from nltk import FreqDist, word_tokenize
from collections import defaultdict

# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

nltk_stopwords = set(stopwords.words('english'))
nltk_stopwords.add('said')
nltk_stopwords.add('mr')
nltk_stopwords.add('us')
nltk_stopwords.add('would')
nltk_stopwords.add('also')


def consolidate_files():
    with open('data.txt', 'w', encoding='utf8') as outfile:
        for label in glb.TEXT_CLASS:
            directory = '%s/%s' % (glb.BASE_DIRECTORY, label)
            for filename in os.listdir(directory):
                full_filename = '%s/%s' % (directory, filename)
                # print(full_filename)
                with open(full_filename, 'rb') as file:
                    text = file.read().decode(errors='replace').replace('\n\n', ' ').replace('\n', '')\
                        .replace(chr(34), '').replace('chr(233)', 'e')
                    outfile.write('%s\t%s\n' % (label, text))


def load_docs():
    docs = []
    with open('data.txt', 'r', encoding='utf8') as datafile:
        for row in datafile:
            col = row.split('\t')
            doc = (col[0], col[1].strip())
            docs.append(doc)
    return docs


def simplify(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text


def frequency_analysis(wall_of_text):
    tokens = defaultdict(list)

    for col in wall_of_text:
        label = col[0]
        text_wall = simplify(col[1])  # remove random symbols and make it all lowercase
        word_list = word_tokenize(text_wall)
        word_list = [i for i in word_list if i not in nltk_stopwords]
        tokens[label].extend(word_list)

    for category_label, category_tokens in tokens.items():
        print(category_label)
        fd = FreqDist(category_tokens)
        print(fd.most_common(20))


def split_data(text_to_classify):
    num_samples = len(text_to_classify)

    # shuffle the data
    random.shuffle(text_to_classify)

    text_training_xtr = []      # training text data
    label_training_ytr = []  # categories for the training text data

    text_testing_xtst = []       # testing text data
    label_testing_ytst = []  # categories for the testing text data

    changeover = int(0.80 * num_samples)

    for i in range(0, changeover):
        text_training_xtr.append(text_to_classify[i][1])
        label_training_ytr.append(text_to_classify[i][0])

    for i in range(changeover, num_samples):
        text_testing_xtst.append(text_to_classify[i][1])
        label_testing_ytst.append(text_to_classify[i][0])

    return text_training_xtr, label_training_ytr, text_testing_xtst, label_testing_ytst


def evaluate_classifier(title, classifier, vectored_text, x_test, y_test):
    x_test_tfidf = vectored_text.transform(x_test)
    y_pred = classifier.predict(x_test_tfidf)

    precision = metrics.precision_score(y_test, y_pred, average='weighted')
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')

    print("Classifier\tSet\t\tPrecision\tRecall\t\tF1")
    print("%s\t%f\t%f\t%f\n" % (title, precision, recall, f1))

    return precision


def train_classifier_naive_bayes(text_to_classify, iterate_features):
    test_precision = 0.0
    num_features = 5
    if iterate_features == 1:
        while test_precision < 0.90:
            # custom function
            text_training_xtr, label_training_ytr, text_testing_xtst, label_testing_ytst = split_data(text_to_classify)
            # library function
            # text_training_xtr, text_testing_xtst, label_training_ytr, label_testing_ytst = \
            #    sklearn.model_selection.train_test_split(text_to_classify[1], text_to_classify[0], test_size=0.20)

            # turn text into vectors in some method - here by count
            text_vectorized = CountVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.5,
                min_df=0.05,
                analyzer='word',
                max_features=num_features)

            # doc-term matrix
            doc_term_matrix = text_vectorized.fit_transform(text_training_xtr, label_training_ytr)

            # train naive bayes
            naive_bayes = MultinomialNB(alpha=1.0, fit_prior='true').fit(doc_term_matrix, label_training_ytr)

            # see how it does
            print("Number of Features: %s" % num_features)
            evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes, text_vectorized,
                                text_training_xtr, label_training_ytr)
            test_precision = evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes,
                                                 text_vectorized, text_testing_xtst, label_testing_ytst)

            num_features = num_features + 10
    else:

        # custom function
        text_training_xtr, label_training_ytr, text_testing_xtst, label_testing_ytst = split_data(text_to_classify)

        # turn text into vectors in some method - here by count
        text_vectorized = CountVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.5,
            min_df=0.05,
            analyzer='word',
            max_features=glb.NUM_FEATURES_NB)

        # doc-term matrix
        doc_term_matrix = text_vectorized.fit_transform(text_training_xtr, label_training_ytr)

        # train naive bayes
        naive_bayes = MultinomialNB(alpha=1.0, fit_prior='true').fit(doc_term_matrix, label_training_ytr)

        # see how it does
        evaluate_classifier("Naive Bayes\tTRAIN\t", naive_bayes, text_vectorized, text_training_xtr, label_training_ytr)
        evaluate_classifier("Naive Bayes\tTEST\t", naive_bayes, text_vectorized, text_testing_xtst, label_testing_ytst)

    # store classifier
    classifier_filename = 'naive_bayes_classifier.pkl'
    pickle.dump(naive_bayes, open(classifier_filename, 'wb'))

    # store vectorizor so we can transform new data
    vectorizer_filename = 'count_vectorizer.pkl'
    pickle.dump(text_vectorized, open(vectorizer_filename, 'wb'))


def use_trained_classifier_nb(wall_of_text):
    # load classifier
    classifier_filename = 'naive_bayes_classifier.pkl'
    neural_net = pickle.load(open(classifier_filename, 'rb'))

    # store vectorizor so we can transform new data
    vectorizer_filename = 'count_vectorizer.pkl'
    text_vectorized = pickle.load(open(vectorizer_filename, 'rb'))

    predicted = neural_net.predict(text_vectorized.transform([wall_of_text]))

    print("Test Text Classification / Naive Bayes: %s" % predicted[0])


def use_trained_classifier_nn(wall_of_text):
    # load classifier
    classifier_filename = 'nn_classifier.pkl'
    neural_net = pickle.load(open(classifier_filename, 'rb'))

    # store vectorizor so we can transform new data
    vectorizer_filename = 'count_vectorizer_nn.pkl'
    text_vectorized = pickle.load(open(vectorizer_filename, 'rb'))

    predicted = neural_net.predict(text_vectorized.transform([wall_of_text]))

    print("Test Text Classification / Neural Net: %s" % predicted[0])


def train_classifier_multilayer_perceptron(text_to_classify):

    # custom function
    text_training_xtr, label_training_ytr, text_testing_xtst, label_testing_ytst = split_data(text_to_classify)

    # turn text into vectors in some method - here by count
    text_vectorized = CountVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_df=0.30,
        min_df=0.05,
        analyzer='word',
        max_features=glb.NUM_FEATURES_NN)

    # doc-term matrix
    doc_term_matrix = text_vectorized.fit_transform(text_training_xtr, label_training_ytr)

    # train naive bayes
    neural_net = MLPClassifier(random_state=1, max_iter=10000).fit(doc_term_matrix, label_training_ytr)

    # see how it does
    evaluate_classifier("Neural Net\tTRAIN\t", neural_net, text_vectorized, text_training_xtr, label_training_ytr)
    evaluate_classifier(
        "Neural Net\tTEST\t", neural_net, text_vectorized, text_testing_xtst, label_testing_ytst)

    # store classifier
    classifier_filename = 'nn_classifier.pkl'
    pickle.dump(neural_net, open(classifier_filename, 'wb'))

    # store vectorizor so we can transform new data
    vectorizer_filename = 'count_vectorizer_nn.pkl'
    pickle.dump(text_vectorized, open(vectorizer_filename, 'wb'))
