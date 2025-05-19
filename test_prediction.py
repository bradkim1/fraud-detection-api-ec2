import requests
import json
import random
from datetime import datetime

# Get current day of week (0-6) and hour (0-23)
current_datetime = datetime.now()
current_day = current_datetime.weekday()  # Monday is 0, Sunday is 6
current_hour = current_datetime.hour      # 0-23

# Initialize a dictionary with default values for all required features
required_features = {
    # Time features that were missing
    'day': current_day,  # Add day of week
    'hour': current_hour,  # Add hour of day
    
    # Numeric features
    'TransactionAmt': 100.0,
    'card1': 1234,
    'card2': 123,
    'card3': 150,
    'card5': 100,
    'addr1': 300,
    'addr2': 87,
    'dist1': 10,
    'dist2': 10,
    
    # C values (numeric)
    'C1': 1.0, 'C2': 1.0, 'C3': 1.0, 'C4': 1.0, 'C5': 1.0,
    'C6': 1.0, 'C7': 1.0, 'C8': 1.0, 'C9': 1.0, 'C10': 1.0,
    'C11': 1.0, 'C12': 1.0, 'C13': 1.0, 'C14': 1.0,
    
    # D values (numeric)
    'D1': 0, 'D2': 0, 'D3': 0, 'D4': 0, 'D5': 0,
    'D6': 0, 'D7': 0, 'D8': 0, 'D9': 0, 'D10': 0,
    'D11': 0, 'D12': 0, 'D13': 0, 'D14': 0, 'D15': 0,
    
    # Categorical features
    'ProductCD': 'C',
    'card4': 'visa',
    'card6': 'debit',
    'P_emaildomain': 'gmail.com',
    'R_emaildomain': 'gmail.com',
    'M1': 'T', 'M2': 'T', 'M3': 'T', 'M4': 'M', 'M5': 'F',
    'M6': 'T', 'M7': 'T', 'M8': 'T', 'M9': 'T',
    
    # Add some V values (there are many of these)
    'V1': 1.0, 'V2': 1.0, 'V3': 1.0, 'V4': 1.0, 'V5': 1.0,
    'V6': 1.0, 'V7': 1.0, 'V8': 1.0, 'V9': 1.0, 'V10': 1.0,
    'V11': 1.0, 'V12': 1.0, 'V13': 1.0, 'V14': 1.0
}

# Fill in remaining V values (there could be many)
for i in range(15, 340):  # Assuming V goes up to V339
    feature_name = f'V{i}'
    if feature_name not in required_features:
        required_features[feature_name] = 0.0

# We also need to add derived features that might be expected
required_features['C_sum'] = sum(required_features[f'C{i}'] for i in range(1, 15))
required_features['D_missing'] = 0  # Assuming no missing D values
required_features['V_mean'] = sum(required_features[f'V{i}'] for i in range(1, 15)) / 14

# Send the request
url = "http://localhost:8010/predict"
headers = {"Content-Type": "application/json"}
payload = {"data": required_features}

print("Sending test prediction request...")
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Print the response
print(f"Status code: {response.status_code}")
if response.status_code == 200:
    print("Success!")
    print(json.dumps(response.json(), indent=2))
else:
    print("Error:")
    print(response.text)
