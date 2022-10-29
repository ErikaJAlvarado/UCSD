from distutils.log import set_verbosity
from telnetlib import theNULL
import pandas as pd
import os
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fast_ml.model_development import train_valid_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import argparse
import logging

def data_load():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    os.chdir("../data/raw_data")
    cwd = os.getcwd()

    train_file_path = os.path.join(cwd,"train.csv")
    train_df = pd.read_csv(train_file_path, encoding='utf-8')
    return(train_df)

def pre_processing_data(df):
    """
    This function process and clean raw data collected from tweeter to be used during model training
    Inputs:
        :df     :   raw data dataframe
        :type   :   data frame

    Returns     :   A dataframe with data pre-processed to be used for training
    """

    if df is None:
        raise ValueError("Training data does not exist")    

    train_df = df

    #1. Fixing Format on the training data: Removing carrier return in all columns.
    train_df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r", "\r"], value=["","",""], regex=True, inplace=True)

    #2. Copy df into a file to clean data line by line.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    os.chdir("../data/processed_data")
    cwd = os.getcwd()

    train_file_path = os.path.join(cwd,"data_temp.csv")
    train_df.to_csv(train_file_path, index = False, header = False)    

    #3. Formatting data to get all values into their corresponding columns
    lst = []
    dic = {}
    count = 0

    id_pattern = "^.+?,"                                #anything before the first comma
    keyword_pattern = ",.*?,"                           #anything between the first and second comma.  It can be null
    location_pattern_1 = ",\".+?\",{1}"                 #first quoted value between commas
    location_pattern_2 = ",.*?,(.*?),"                  #grouping: second value between commas.  It can be null

    with open (train_file_path, "r") as f:
        for line in f:

            # removing commas at the end of the row.
            line = line.strip("\n")
            line = line.strip(",")
            line = line.strip()        

            #Getting Id
            id_search = re.search(id_pattern, line)        
            id = id_search.group()[:-1]                
            
            #Getting keyword
            keyword_search = re.search(keyword_pattern,line)
            keyword = keyword_search.group()[1:-1]
            
            if re.search(location_pattern_1, line) != None:    
                #Getting location
                location_search = re.search(location_pattern_1, line)
                location = location_search.group()[1:-1]                        
                
            else:    
                #Getting location
                location_search = re.search(location_pattern_2, line)
                location = location_search.group(1)        
                        
            #Getting target                             #anything after the last comma
            target = line.rsplit(',',1)[1]
            
            #Getting text
            before_text = id + "," + keyword + "," + location + ","
            after_text = ","+target
            text = line.replace(before_text,"")         #removed id, keyword, loaction from the orininal string
            text = text.replace(after_text,"")          #removed target. Text is what is left - with commas, quotes, anything so we keep the text as complete as possible.
                            
            #Getting data into a dataframe
            dic = {}
            dic["id"] = id
            dic["keyword"]=keyword
            dic["location"]= location
            dic["text"]=text
            dic["target"] = target
            lst.append(dic)
        
    train_df = pd.DataFrame(lst)
    train_df["target"] = train_df["target"].astype(float)
    train_df["target"] = train_df["target"].astype(int)
    return train_df


def get_hashtags(text):
    """
    This function find all hashtags within a text and remove #
    Inputs:
        :text   :   text
        :type   :   string

    Returns     :   A string with all hashtags without #
    """


    string = ""
    hashtag_pattern = r"(#[^ ]+)"
    hashtag = re.findall(hashtag_pattern,text)
    
    for element in hashtag:
        element = element.replace("#","")
        string += element + " "
        string = string.strip()
    
    return string


def get_mentions(text):
    string = ""
    mentions_pattern = r"(@[^ ]+)"
    mentions = re.findall(mentions_pattern,text)
    
    for element in mentions:
        element = element.replace("@","")
        string += element + " "
        string = string.strip()
    
    return string

def remove_emoji(string):
    """
    This function removes all emojis from a string
    Inputs:
        :string :   string to remove emojis
        :type   :   string

    Returns     :   A string with no emojis
    """

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r" ", string)    


def cleaning_text(text):
    """
    This function remove url, html, numbers, punctuation, stop words and do stemming
    Inputs:
        :text   :   text
        :type   :   string

    Returns     :   Clean string.
    """

    url_pattern = "http\S*"
    numb_word_pattern = "\d"
    html_pattern = "<.*?>"
    punctuation_pattern = string.punctuation
    
    #remove urls
    url_clean = re.sub(url_pattern," ",text)
    
    #remove html code     
    html_clean = re.sub(html_pattern," ", url_clean)    
       
    #remove numbers
    numbers_clean = re.sub(numb_word_pattern," ",html_clean)
    
    #remove punctuation
    punctuation_clean = re.sub("[{}]".format(punctuation_pattern)," ",numbers_clean)
    
    #tokenization
    tokens = word_tokenize(punctuation_clean)
    
    #removing stop words
    stop_words = stopwords.words('english')
    stop_words_clean = [word.strip() for word in tokens if word.strip() not in stop_words]
    
    #Stemming
    stemmer = nltk.stem.PorterStemmer()
    stemmer_clean = [stemmer.stem(word) for word in stop_words_clean]
    stemmer_clean_str = " ".join(stemmer_clean)        
        
    return stemmer_clean_str

def missing_data(df):

    train_df = df

    #Dropping rows with no keywords
    train_df.dropna(subset=["clean_keyword"], inplace=True)
    return train_df  


def cleaning_hex(text):
    """
    This function remove all hex characters from text
    Inputs:
        :text   :   text
        :type   :   string

    Returns     :   String with no hex characters
    """

    clean_hex = re.sub(r'[^ -~].*'.format(string.punctuation)," ", text)
    return clean_hex

def cleaning_amp(text):
    """
    This function remove & from text
    Inputs:
        :text   :   text
        :type   :   string

    Returns     :   A string with no &
    """

    clean = re.sub(r'\samp\s'," ", text)
    return clean    


def cleaning_data(df):

    """
    This function cleans data not only one string but the whole dataset.
    Removes emojis, punctuation, etc and combines keyword + text
    Inputs:
        :text   :   dataset
        :type   :   dataframe

    Returns     :   Pre-processed and clean dataset.
    """



    if df is None:
        raise ValueError("Training data does not exist")    

    train_df = df
    
    # 1. Text to lower case.
    train_df["text"] = train_df["text"].str.lower()
    train_df["keyword"] = train_df["keyword"].str.lower()

    # 2. Getting hashtag from text.
    train_df["hashtags"] = train_df["text"].apply(lambda x : get_hashtags(x))
    
    # 3. Getting mentions from text.
    train_df["mentions"] = train_df["text"].apply(lambda x : get_mentions(x))

    # 4. Removing emojis
    train_df["clean_emojis"] =  train_df["text"].map(lambda x: remove_emoji(x))

    # 5. Cleaning text
    train_df["clean_text"] = train_df["clean_emojis"].apply(lambda x: cleaning_text(x))

    # 6. Cleaning keyword
    train_df["clean_keyword"] = train_df["keyword"].apply(lambda x: cleaning_text(x))
    
    # 7. Dropping missing data
    train_df = missing_data(train_df)

    # 8. keyword + text    
    train_df["keyword_text"] = train_df["clean_keyword"]+" "+train_df["clean_text"]
    
    # 9. Cleaning hex
    train_df["clean_hex"] = train_df["keyword_text"].apply(lambda x: cleaning_hex(x))

    # 10. Removing amp
    train_df["clean_data"] = train_df["clean_hex"].apply(lambda x: cleaning_amp(x))
    
    data_df = train_df[["clean_data","target"]]
    data_df.rename(columns={"clean_data":"text"}, inplace=True)    
    
    return data_df


def saving_clean_data(df):
        
    if df is None:
        raise ValueError("Training data does not exist")    

    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    os.chdir("../data/processed_data")
    cwd = os.getcwd()
    data_file = os.path.join(cwd,"clean_data.csv")

    df.to_csv(data_file)


def visualization(df):
    
    train_df = df

    emerg_df = train_df.loc[train_df["target"]==1,:]["text"]
    disaster_text_str = " ".join(emerg_df)
    wordcloud = WordCloud().generate(disaster_text_str)
    plt.figure(figsize = (8,8))
    plt.imshow(wordcloud)
    plt.show()


def decoding(text):

    # Loading tokenizer
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)
    tokenizer_path = os.path.join("../tokenizer","tokenizer.pickle")
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    word_index = tokenizer.word_index    
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    return " ".join([reverse_word_index.get(i, "na") for i in text])


def get_severity(x):
    """
    This function classifies in 0,1 and 2 all predictions.
    If prediction is [0-0.5] it returns 0
    If prediction is [0.51-0.75] it returns 1
    If prediction is [0.75-1] it returns 2

    Inputs:
        :x      :   model predicton
        :type   :   number

    Returns     :   A number that classifies the disaster in no-disaster (0) , low probability of being a disaster(1), hight probability  (2)
    """

    try:
        if x is None:
            raise ValueError("Please enter an argument to evaluate severity")

        if not(isinstance(x,int) or isinstance(x,float)):
            raise ValueError("The input needs to be a integer or float")
        
        if x < 0:
            raise ValueError("The input needs to be a positive integer or float")


        if x >= 0 and x <= 0.5:
            severity=0
        elif x>0.5 and x<0.75:
            severity=1
        elif x>=0.75 and x<=1:
            severity=2
    
    except ValueError as err:
        print(str(err))
        raise
    else: 
        return severity


def model(df):
    """
    This function using the clean and processed data and train the model (LSTM)

    Inputs:
        :df     :   clean data
        :type   :   dataframe

    Returns     :   Model, tokenizer and evaluation
    """

    if df is None:
        raise ValueError("Data does not exist to train the model")    

    data = df    
    
    # 1. Splitting data (train, validation and test)    
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_valid_test_split(data, target = 'target',train_size=0.8, valid_size=0.1, test_size=0.1,
                                                                           method = "random", random_state = 17)

    # 2. Saving data for validation
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df  = pd.concat([X_test, y_test], axis=1)
    
    train_file = os.path.join("../data/training_data","train.csv")
    valid_file = os.path.join("../data/training_data","valid.csv")
    test_file = os.path.join("../data/training_data","test.csv")

    train_df.to_csv(train_file)
    valid_df.to_csv(valid_file)
    test_df.to_csv(test_file)

    # 3. Encoding and Padding
    cv = CountVectorizer()
    X = cv.fit_transform(data["text"])
    vocab_size = len(cv.vocabulary_)

    # Tokenizer : assign an index to each word in the vocabulary
    num_words = vocab_size + 1 # +1 for oov
    tokenizer = Tokenizer(num_words=num_words, oov_token = True)
    tokenizer.fit_on_texts(X_train["text"])
    
    # Saving the tokenizer to use it later on predictions
    tokenizer_path = os.path.join("../tokenizer","tokenizer.pickle")
    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Max number of words per tweet (text)
    data["word_count"]  = data["text"].apply(lambda n : len(n.split()))
    max_length          = data["word_count"].max()
    print(max_length)

    # Training: Text to sequence and padding - replace words by index
    X_train_sequences   = tokenizer.texts_to_sequences(X_train["text"])
    X_train_padded      = pad_sequences(X_train_sequences, maxlen=max_length, padding="post", truncating="post")

    # Validation:  Text to sequence and padding
    X_valid_sequences = tokenizer.texts_to_sequences(X_valid["text"])
    X_valid_padded = pad_sequences(X_valid_sequences, maxlen=max_length, padding="post", truncating="post")

    # Test: Text to sequence and padding
    X_test_sequences = tokenizer.texts_to_sequences(X_test["text"])
    X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding="post", truncating="post")

    # 4. Model
    model = Sequential()

    model.add(Embedding(input_dim = num_words, output_dim=32, input_length=max_length))
    model.add(LSTM(64, dropout=0.1))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=3e-4)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        X_train_padded, y_train, epochs=20, validation_data=(X_valid_padded, y_valid),
    )
    
    # 4. Save model
    model_path = os.path.join("../model/td_model")
    model.save(model_path)

    # 5. Evaluation
    loss, accuracy = model.evaluate(X_test_padded,y_test,verbose = 0)    
    print("loss = {} ; accuracy = {}".format(loss, accuracy))


def prediction(model_path, tokenizer_path, data_path, prediction_path):
    """
    This function use the model and tokenizer to identify whether a tweet is a emergency disaster or not. 

    Inputs:
        :model_path         :   model path
        :tokenizer_path     :   tokenizer path
        :data_path          :   data path
        :prediction_path    :   path were final data set will be saved

    Returns     :   A dataframe with all tweets classified as disaster)
    """
        
    if model_path is None:
        raise ValueError("Model does not exist") 

    if tokenizer_path is None:
        raise ValueError("Tokenizer does not exist") 

    if data_path is None:
        raise ValueError("Tweeter data to predict does not exist") 

    if prediction_path is None:
        raise ValueError("Please enter a location to save the output prediction") 

    # Loading model
    model = load_model(model_path)

    # Loading tokenizer
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    # Loading data
    data        = pd.read_csv(data_path)
    
    data_sequences      = tokenizer.texts_to_sequences(data["text"])
    data_padded         = pad_sequences(data_sequences, maxlen=24, padding="post", truncating="post")
    prediction          = model.predict(data_padded)
    prediction          = np.squeeze(prediction) # reshaping from ndim to 1dim
    
    # Adding predictions to the original data
    prediction_series   = pd.Series(prediction)    
    disaster_df = pd.concat([data, prediction_series], axis=1)
    disaster_df.rename(columns={disaster_df.columns[4]:"disaster_pred"},inplace=True)
    
    # Classifying predictions in non-disaster, low prob and high probability of disaster
    disaster_df["severity"]= disaster_df["disaster_pred"].apply(lambda n: get_severity(n))    

    # Discard non-disaster tweets
    disaster_df = disaster_df.loc[disaster_df["severity"]>0,:]

    # Save output prediction data
    disaster_df.to_csv(prediction_path)


def process():
    """
    This function gets the raw data, processed, clean it and save it.

    Returns     :   A dataframe with all tweets classified as disaster)
    """

    raw_data_df = data_load()
    proc_data_df = pre_processing_data(raw_data_df)
    clean_data_df = cleaning_data(proc_data_df)
    saving_clean_data(clean_data_df)


def process_args():
    """
    This function creates the mode argument that will be required to run this model.
    mode = process, process funtion runs
    mode = model, model function runs
    mode = prediction, runs prediction functon.
    
    """


    parser          = argparse.ArgumentParser(description='Collects, Process and model data for Disaster Tweeter classification')
    parser.add_argument('--mode', type = str, metavar = '<MODE>',
                    help='Enter: process, model or prediction',
                    required = True)        
    args            = parser.parse_args()
    return args



if (__name__ == "__main__") :

    args        = process_args()
    mode        = args.mode

    script_dir = os.path.dirname(os.path.realpath(__file__))    
    os.chdir(script_dir)

    #Create and configure logger
    logs_path = os.path.join("../logs/","tweeter_disaster.logs")
    log_format = "%(levelname)s,%(asctime)s,%(message)s"
    logging.basicConfig(filename = logs_path, level= logging.INFO, format = log_format, filemode = "a", datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()


    if mode=="process":        
        process()        
        errmsg = "Data preprocessing completed successfully"
        logger.info(errmsg)

    if mode=="model":        
        data_file_path = os.path.join("../data/processed_data","clean_data.csv")
        data = pd.read_csv(data_file_path, encoding='utf-8')    

        model(data)
        errmsg = "Data model completed successfully"
        logger.info(errmsg)

    if mode=="prediction":
              
        # model path
        model_path = os.path.join("../model/td_model")

        # tokenizer path
        tokenizer_path = os.path.join("../tokenizer","tokenizer.pickle")

        # tweeter data path
        data_path   = os.path.join("../data/tweeter_data","tweeter.csv")        

        # output prediction data path
        prediction_path = os.path.join("../data/prediction","disaster_prediction.csv")
        prediction (model_path, tokenizer_path, data_path, prediction_path)

        errmsg = "Disaster classification completed successfully"
        logger.info(errmsg)
