How to run it

To run the API 
-------------
API returns the list of tweeter classified as emergencies.

1. For design, please look at any of the docs below: production_repo/API_design.pdf or any of the interactive html API_design% folders
	production_repo/docs/API_design.pdf
	production_repo/API/API_design_html-documentation-generated/index.html	
	
2. For instructions on how to run:
	production_repo/docs/How_to_use_Emergency_tweets API.ipynb
	
For code documentation
----------------------
	/production_repo/docs/_build/index.html

	Same script is used for:	
		data load
		data pre-processing
		data modeling
		prediction
		
	Script requires an argument "mode" as it's specified below to process/clean the data, train the model and make predictions.
		tweeter_disaster.py --mode process
		tweeter_disaster.py --mode model
		tweeter_disaster.py --mode prediction


For predictions
---------------
1. Make sure new data from tweeter is inserted into the directory production_repo/data/tweeter_data/tweeter.csv Otherwise the script will send an error.
2. Run python /production_repo/code/tweeter_disaster.py --mode prediction
3. Find the tweets classified as emergency at production_repo/data/prediction/disaster_prediction.csv


For retraining
---------------
1. Make sure data is in /production_repo/data/raw_data/train.csv
2. Run python /nlp_project/code/tweeter_disaster.py --mode process
3. Make sure data has been created into /production_repo/data/processed_data/clean_data.csv (Clean and formatted data is placed here)
4. Run python /production_repo/code/tweeter_disaster.py --mode model
5. For validation training and validation data creating during the model is saved it in /nlp_project/data/training_data
	train.csv
	valid.csv
	test.csv
6. Model is saved under /production_repo//model
7. Tokenizer is saved under /production_repo//tokenizer

