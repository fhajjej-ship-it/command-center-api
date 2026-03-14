import requests
import json
import sys

def test_stream():
    url = "http://127.0.0.1:8000/api/v1/stream/my-pc-is-acting-super-sluggish-and-hot"
    print(f"Connecting to {url}...")
    try:
        with requests.get(url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]
                        try:
                            data = json.loads(data_str)
                            print(f"[{data.get('agent', 'Unknown')}] {data.get('status', '')}: {data.get('action', '')}")
                        except json.JSONDecodeError:
                            print(f"Raw data: {data_str}")
                    else:
                        print(decoded)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_stream()
