from flask import Flask, render_template
import requests
app = Flask(__name__)
NASA_API = "hpDUugVbKilEQl3SLpViHw4hNFOz0iHq313KZY3i"
APOD_URL = "https://api.nasa.gov/planetary/apod"

@app.route("/", methods=["GET"])
def index():
    par = {"api_key" : NASA_API}
    response = requests.get(APOD_URL, params=par)
    apod_data = response.json()
    return render_template("index.html", apod = apod_data)
if __name__ == "__main__":
    app.run(debug=False)