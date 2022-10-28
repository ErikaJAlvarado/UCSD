from flask import Flask
import os

# Flask
app = Flask(__name__)

@app.route('/twitter_emergency_api/<input_text>')
def twitter_emergency_api(input_text):
    output_text = "Terremotito: "+input_text
    return output_text

if __name__== "__main__":
    app.run(debug=True,host="0.0.0.0",port=int(os.environ.get("PORT",5001)))

