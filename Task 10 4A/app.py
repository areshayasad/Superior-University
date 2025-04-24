from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
database = {
    "timings":"The library is open from 8 AM to 8 PM, Monday to Saturday.",
    "location":"The library is located on the 2nd floor of the Academic Building.",
    "membership":"To get a membership, fill the registration form at the front desk.",
    "rules":"No eating or drinking is allowed. Maintain silence at all times.",
    "books":"You can borrow up to 3 books for 14 days. Renewals allowed once."
}
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.form["msg"].lower()
    response = "Sorry, I didn't understand that. Try asking about timings, location, membership, rules, or books."
    for key in database:
        if key in user_input:
            response = database[key]
            break
    return jsonify({"response": response})
if __name__ == "__main__":
    app.run(debug=True)
