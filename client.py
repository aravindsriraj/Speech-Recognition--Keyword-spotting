from email.policy import strict
from urllib import response
import requests
import json
URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = r"C:\Users\MYPC\Desktop\coding\Speech recognition tensorflow\right.wav"

if __name__=="__main__":
    audio_file = open(TEST_AUDIO_FILE_PATH,'rb')
    values = {"file":(TEST_AUDIO_FILE_PATH,audio_file,"audio/wav")}
    response = requests.post(URL,files=values)
    data = json.load(response.json(),strict=False)

    print(f"Predicted Keyword is {data['keyword']}")