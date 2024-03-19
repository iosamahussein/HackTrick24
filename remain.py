import requests
import json

end_point = "http://13.53.169.72:5000/attempts/student"
params = {
    "teamId": "Kps2iU3",
}
params = json.dumps(params)
response = requests.post(end_point, data=params, headers={"Content-Type": "application/json"})


response_data = response.json()

print(f"Remaining eagle: {response_data['remaining_eagle_attempts']}")
print(f"Remaining fox: {response_data['remaining_fox_attempts']}")

print(response.text)
print(response.status_code)