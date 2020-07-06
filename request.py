  
import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'SepalLengthCm':2, 'SepalWidthCm':9, 'PetalLengthCm':6, 'PetalWidthCm':2})

print(r.json())