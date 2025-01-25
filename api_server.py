from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import load_llm, summarize_text
from PyPDF2 import PdfReader
import io
import argparse

app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM instance
llm = None

@app.post("/summarize/")
async def summarize_pdf(
    pdf_file: UploadFile = File(...),
    max_length: int = 500
):
    try:
        # Extract text from PDF
        contents = await pdf_file.read()
        reader = PdfReader(io.BytesIO(contents))
        text = "\n".join([page.extract_text() for page in reader.pages])

        # Generate summary
        summary = summarize_text(llm, text, max_length)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api_server(model_path: str):
    global llm
    llm = load_llm(model_path)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
    args = parser.parse_args()
    start_api_server(args.model)