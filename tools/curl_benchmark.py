import requests
import json
import time

# Define the URL and headers
url = "http://localhost:8080/completion"
headers = {
    "Content-Type": "application/json"
}

# Define the data payload
data = {
    "prompt": "Im stressed, what should i do?",
    "n_predict": 128
}

# Record the start time
start_time = time.time()

# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))

# Record the end time
end_time = time.time()

# Calculate the overall time taken
overall_time = end_time - start_time

# Check if the request was successful
if response.status_code == 200:
    # Print the response from the server
    print("Response from server:")
    print(response.json())
else:
    # Print the error
    print(f"Request failed with status code {response.status_code}")
    print(response.text)

# Print the overall time taken
print(f"Overall time taken for the request: {overall_time:.2f} seconds")
