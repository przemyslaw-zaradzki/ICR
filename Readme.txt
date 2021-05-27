python -m venv venv     # utworzenie środowiska wirtualnego
venv\Scripts\activate   # aktywacja

#pip install flask flask-wtf flask-uploads      # flask-uploads nie działa
pip install flask flask-wtf Flask-Reuploaded

set FLASK_ENV=development       # ustawienie zmiennych śrdowiskowych
set FLASK_APP=app.py
flask run