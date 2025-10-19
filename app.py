from flask import Flask, render_template, request
import sqlite3
import pickle
from utils.data_processing import preprocess_input

app = Flask(__name__)

# Load AI model
model = pickle.load(open('models/diabetes_model.pkl', 'rb'))

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Add patient data
@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']

        # Save to SQLite
        conn = sqlite3.connect('database/patients.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (name, age, glucose, blood_pressure)
            VALUES (?, ?, ?, ?)
        ''', (name, age, glucose, blood_pressure))
        conn.commit()
        conn.close()

        # AI prediction
        input_data = preprocess_input([age, glucose, blood_pressure])
        prediction = model.predict([input_data])[0]

        return render_template('result.html', prediction=prediction)
    return render_template('add_patient.html')

# View all patients
@app.route('/patients')
def patients():
    conn = sqlite3.connect('database/patients.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM patients')
    data = cursor.fetchall()
    conn.close()
    return render_template('view_patients.html', patients=data)
# Edit patient
@app.route('/edit_patient/<int:id>', methods=['GET', 'POST'])
def edit_patient(id):
    conn = sqlite3.connect('database/patients.db')
    cursor = conn.cursor()
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        glucose = request.form['glucose']
        blood_pressure = request.form['blood_pressure']
        cursor.execute('''
            UPDATE patients
            SET name=?, age=?, glucose=?, blood_pressure=?
            WHERE id=?
        ''', (name, age, glucose, blood_pressure, id))
        conn.commit()
        conn.close()
        return render_template('result.html', prediction=None, message="Patient details updated successfully!")
    cursor.execute('SELECT * FROM patients WHERE id=?', (id,))
    patient = cursor.fetchone()
    conn.close()
    return render_template('edit_patient.html', patient=patient)

# Delete patient
@app.route('/delete_patient/<int:id>')
def delete_patient(id):
    conn = sqlite3.connect('database/patients.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM patients WHERE id=?', (id,))
    conn.commit()
    conn.close()
    return render_template('result.html', prediction=None, message="Patient deleted successfully!")


if __name__ == '__main__':
    app.run(debug=True)
