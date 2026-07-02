from flask import Flask, render_template, request, redirect
import sqlite3
import pickle
import os
from utils.data_processing import preprocess_input

app = Flask(__name__)

# ---------------- LOAD MODEL ---------------- #

model = pickle.load(open("models/diabetes_model.pkl", "rb"))

# ---------------- CREATE DATABASE ---------------- #

os.makedirs("database", exist_ok=True)

conn = sqlite3.connect("database/patients.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS patients(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    glucose REAL,
    blood_pressure REAL
)
""")

conn.commit()
conn.close()


# ---------------- HOME ---------------- #

@app.route("/")
def index():
    return render_template("index.html")


# ---------------- ADD PATIENT ---------------- #

@app.route("/add_patient", methods=["GET", "POST"])
def add_patient():

    if request.method == "POST":

        name = request.form["name"]
        age = int(request.form["age"])
        glucose = float(request.form["glucose"])
        blood_pressure = float(request.form["blood_pressure"])

        conn = sqlite3.connect("database/patients.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO patients(name, age, glucose, blood_pressure)
        VALUES(?,?,?,?)
        """, (name, age, glucose, blood_pressure))

        conn.commit()
        conn.close()

        input_data = preprocess_input([age, glucose, blood_pressure])

        prediction = model.predict([input_data])[0]

        if prediction == 1:
            prediction = "High Risk"
        else:
            prediction = "Low Risk"

        return render_template(
            "result.html",
            prediction=prediction
        )

    return render_template("add_patient.html")


# ---------------- VIEW PATIENTS ---------------- #

@app.route("/view_patients")
def view_patients():

    conn = sqlite3.connect("database/patients.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patients")

    patients = cursor.fetchall()

    conn.close()

    return render_template(
        "view_patients.html",
        patients=patients
    )


# ---------------- EDIT PATIENT ---------------- #

@app.route("/edit_patient/<int:id>", methods=["GET", "POST"])
def edit_patient(id):

    conn = sqlite3.connect("database/patients.db")
    cursor = conn.cursor()

    if request.method == "POST":

        name = request.form["name"]
        age = request.form["age"]
        glucose = request.form["glucose"]
        blood_pressure = request.form["blood_pressure"]

        cursor.execute("""
        UPDATE patients
        SET
        name=?,
        age=?,
        glucose=?,
        blood_pressure=?
        WHERE id=?
        """, (name, age, glucose, blood_pressure, id))

        conn.commit()
        conn.close()

        return redirect("/view_patients")

    cursor.execute(
        "SELECT * FROM patients WHERE id=?",
        (id,)
    )

    patient = cursor.fetchone()

    conn.close()

    return render_template(
        "edit_patient.html",
        patient=patient
    )


# ---------------- DELETE PATIENT ---------------- #

@app.route("/delete_patient/<int:id>")
def delete_patient(id):

    conn = sqlite3.connect("database/patients.db")
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM patients WHERE id=?",
        (id,)
    )

    conn.commit()
    conn.close()

    return redirect("/view_patients")


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(debug=True)