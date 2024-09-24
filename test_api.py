from fastapi.testclient import TestClient
from app import app  # assuming your FastAPI app is in app.py

client = TestClient(app)

def test_health_prediction():
    test_data = {
        "datetimeEpo": 1622918421.0,
        "tempmax": 32.0,
        "tempmin": 24.0,
        "temp": 28.0,
        "feelslikemax": 35.0,
        "feelslikemin": 22.0,
        "feelslike": 30.0,
        "dew": 12.0,
        "humidity": 75.0,
        "snow": 0.0,
        "snowdepth": 0.0,
        "windgust": 45.0,
        "solarradiation": 200.0,
        "uvindex": 5.0,
        "sunriseEpoch": 1622883600.0,
        "sunsetEpoch": 1622934000.0,
        "moonphase": 0.5,
        "conditions": "Clear",
        "description": "Sunny with clear skies",
        "icon": "clear-day",
        "source": "weather_api",
        "City": "New York",
        "Temp_Range": 8.0,
        "Heat_Index": 32.0,
        "Severity_Score": 1.5,
        "Day_of_Week": "Monday",
        "Is_Weekend": False,
        "sunset_hour": 19,
        "sunrise_hour": 6,
        "sunrise_minute": 30
    }

    response = client.post("/predict", json=test_data)
    
    # Check if the response is successful (status code 200)
    assert response.status_code == 200

    # Check that the response contains the 'prediction' key
    assert "prediction" in response.json()

    # You can further inspect the prediction value if you have specific expectations
    prediction = response.json()["prediction"]
    assert isinstance(prediction, list)  # Assuming the prediction is returned as a list
