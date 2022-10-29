from lib2to3.pgen2.tokenize import tokenize
from secrets import token_urlsafe
from flask import Flask
import os



# Flask
app = Flask(__name__)

@app.route('/twitter_emergency_api/<word>')
def twitter_emergency_api(word):

    path = '../data/prediction/disaster_prediction.csv'
    load_data(path)

    result = []
    
    
    if word not in words_dict.keys():
        message = "No tweeter was found for that search"
    else:        
        #print(set(words_dict[word]))
        for doc_id in set(words_dict[word]):            
            # list of tupples
            #print(doc_id)
            response = {}
            response["text"] = docs_dict[doc_id][0]            
            response["disaster_prob"] = docs_dict[doc_id][1]
            response["disaster_type"] = docs_dict[doc_id][2]
            #print(response)
            result.append(response)

    #print(result)

    return result


# Mapping words with texts that'll be used within the text search by keyword
def load_data(path):        

    # Data in memory
    global docs_dict
    global words_dict
    docs_dict = {}
    words_dict = {}

    with open(path, 'r') as datafile:

        # reading text line by line
        for line in datafile:
            temp = line.strip().split(",")            

            # Doc dictionary: id = text, class_prob, disaster_flag
            docs_dict[temp[2]] = (temp[3], temp[5], temp[6])

            # splitting text
            tokens = temp[3].split(" ")

            # Creating a word dictionary.  Each key holds a list of docs ids where the word was found.
            for token in tokens:
                if token not in words_dict.keys():
                    words_dict[token] = [temp[2]]
                else:
                    words_dict[token].append(temp[2])

    

if __name__== "__main__":

    data = twitter_emergency_api("fire")
    #print(data)

    #path = './data/disaster_prediction.csv'
    #path = '../data/prediction/disaster_prediction.csv'
    #load_data(path)
    
        
    
    app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",5001)))


