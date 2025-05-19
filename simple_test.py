import requests
import json

# Create a simplified test case with just a few key features
simple_test = {
    "TransactionAmt": 100.0,
    "ProductCD": "C",
    "card1": 1234
}

# Send the request
url = "http://localhost:8015/predict"
headers = {"Content-Type": "application/json"}
payload = {"data": simple_test}

print("Sending simplified test prediction request...")
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print(f"Status code: {response.status_code}")
if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2))
else:
    print("Error:")
    print(response.text)
