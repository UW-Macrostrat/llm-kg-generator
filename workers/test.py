import httpx
import asyncio

RESULT_ENDPOINT = "http://cosmos0003.chtc.wisc.edu:9543/record_run"  # Replace with the actual endpoint

async def post_json_data():
    # Read JSON data from file
    with open('out/out.json', 'r') as file:
        output_json = file.read()

    # Send POST request
    async with httpx.AsyncClient() as client:
        response = await client.post(RESULT_ENDPOINT, headers={"Content-Type": "application/json"}, content=output_json)
        print(response.status_code)
        print(response.text)

# Run the async function
asyncio.run(post_json_data())