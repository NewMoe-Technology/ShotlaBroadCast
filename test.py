import httpx
import asyncio
from time import perf_counter

api: str = "http://localhost:8000/convert"

async def send_request(input_bytes: bytes):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            api,
            files={"WAVBuffer": input_bytes}
        )
        return response.content

async def main():
    input_bytes: bytes = open("./base (2).mp3", "rb").read()
    print(f"Input Length: {len(input_bytes)}")

    start = perf_counter()
    tasks = [send_request(input_bytes) for _ in range(10)]
    responses = await asyncio.gather(*tasks)

    # for i, response_length in enumerate(responses, 1):
    #     print(f"Response {i} Length: {response_length}")
    print(f"Inference cost: {round(perf_counter() - start, 2)} seconds.")
    print(f"Average cost: {round((perf_counter() - start) / len(responses), 2)} seconds.")

    # 保存最后一个响应
    with open("response.wav", "wb") as f:
        f.write(responses[-1])

if __name__ == "__main__":
    asyncio.run(main())

