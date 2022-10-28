How to run it

For predictions
---------------
1. Make sure new data from tweeter is inserted into the directory data/tweeter_data/tweeter.csv
2. Run python /nlp_project/code/tweeter_disaster.py --mode prediction
3. Only tweeter classified as disaster are saved under nlp_project/data/prediction/disaster_prediction.csv


For retraining
---------------
1. Make sure data is in /nlp_project/data/raw_data/train.csv
2. Run python /nlp_project/code/tweeter_disaster.py --mode process
3. Make sure data has been created into /nlp_project/data/processed_data/clean_data.csv (Clean and formatted data is placed here)
4. Run python /nlp_project/code/tweeter_disaster.py --mode model
5. For validation training and validation data creating during the model is saved it in /nlp_project/data/training_data
	train.csv
	valid.csv
	test.csv
6. Model is saved under /nlp_project/model
7. Tokenizer is saved under /nlp_project/tokenizer

