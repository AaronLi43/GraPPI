# test_api.py
import requests

# Replace with the ngrok URL provided by Colab
NGROK_URL = "https://6cf3-34-23-46-115.ngrok-free.app/"  

def test_api():
    # Test data
    data = {
        "start_protein": "MARK4",
        "num_graphs": 1,
        "query": "Find protein paths that the end protein phosphorylates MARK4"
    }

    # First test the health endpoint
    try:
        health_response = requests.get(f"{NGROK_URL}/health")
        print("Health check status:", health_response.status_code)
        print("Health check response:", health_response.json())
    except Exception as e:
        print("Health check failed:", str(e))
        return

    # Then test the analyze endpoint
    try:
        response = requests.post(f"{NGROK_URL}/analyze", json=data)
        print("\nAnalyze endpoint:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSuccess! Response:")
            print(response.json())
        else:
            print("\nError:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Colab server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_api()