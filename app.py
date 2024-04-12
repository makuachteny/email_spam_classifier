from flask import Flask, render_template, request
import pickle

# Create an instance of the Flask class
app = Flask(__name__)

# Load the models
cv = pickle.load(open('models/cv.pkl', 'rb'))  # Count Vectorizer
clf = pickle.load(open('models/clf.pkl', 'rb'))  # Classifier Model


@app.route('/')
def home():
    # Added default values for email and prediction variables
    return render_template('index.html', email='', prediction='')


@app.route('/classify', methods=['POST'])
def classify():
    email = request.form['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    # Passes email variable back to the template
    return render_template('index.html', email=email, prediction=prediction)

# Get the style


@app.route('/style.css')
def send_css():
    return app.send_static_file('style.css')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
