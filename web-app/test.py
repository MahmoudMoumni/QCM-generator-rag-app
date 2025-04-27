import requests
import json

# Your base URL
base_url = "https://ee2f-2001-a18-a-1fe9-963f-9aa7-2952-dab.ngrok-free.app"


url = f"{base_url}/generate_quizz"




# Metadata to send
metadata_payload = {
    "num_questions": 5,
    "question_type": "1",
    "keywords": [],
    "selected_doc_id": 4
}

# Send the POST request with form data
response = requests.post(
    url,
    data={
        "metadata": json.dumps(metadata_payload)
    }
)

# Print the response
if response.status_code == 200:
    print("Quiz generated successfully:")
    print(response.json())
else:
    print(f"Failed to generate quiz. Status code: {response.status_code}")
    print(response.text)

