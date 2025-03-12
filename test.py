import requests
from time import perf_counter

api:str = "http://localhost:8000/convert"

input_bytes:bytes = open("./base (2).mp3", "rb").read()

print(f"Input Length: {len(input_bytes)}")

start = perf_counter()
response:bytes = requests.post(
    api,
    files = {
        "WAVBuffer": input_bytes
    }
)

print(f"Response Legnth: {len(response.content)},inference cost: {round(perf_counter() - start,2)} seconds.")

