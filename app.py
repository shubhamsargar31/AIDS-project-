from flask import Flask, render_template, request, url_for
import os
from predict import predict_skin

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the file is present
        if 'image' not in request.files:
            return render_template("index.html", error="No file selected")
        
        file = request.files["image"]
        if file.filename == '':
            return render_template("index.html", error="No file selected")

        if file:
            # Save the file
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Prediction with new function
            result, cancer_type, confidence = predict_skin(filepath)

            # Pass all variables explicitly to the template
            return render_template(
                "index.html", 
                result=result, 
                cancer_type=cancer_type, 
                confidence=round(confidence, 2),
                uploaded_image=filename
            )

    return render_template("index.html", result=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
