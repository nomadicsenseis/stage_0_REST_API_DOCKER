import requests

url = 'http://pretrained_vit_HAM10000_cpu.pkl:5000/predict'
file_path = 'path/to/your/image.jpg'

with open(file_path, 'rb') as file:
    response = requests.post(url, files={'file': file})

print(response.json())
