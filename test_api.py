from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test API is working!"}

if __name__ == "__main__":
    print("Starting test API on port 8005...")
    uvicorn.run("test_api:app", host="0.0.0.0", port=8005)
