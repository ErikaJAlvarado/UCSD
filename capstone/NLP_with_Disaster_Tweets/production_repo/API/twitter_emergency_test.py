import requests

URL = "http://127.0.0.1:5001/twitter_emergency_api/"

# sending get request and saving the response as response object
input_text = "fire"
FINAL_URL = URL + input_text
print('sending:', FINAL_URL)
r = requests.get(url = FINAL_URL)

# report results
if str(r) == '<Response [200]>':    
    print('success:', r.text)
else:
    print('error!', str(r))


input_text = 'erika'
FINAL_URL = URL + input_text
print('sending:', FINAL_URL)
r = requests.get(url = FINAL_URL)

# report results
print(r)
if str(r) == '<Response [200]>':
    print('success:', r.text)
else:
    print('error!', str(r))
