from llama_cpp import Llama
from typing import Optional

def load_llm(model_path: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"):
    """Load a local LLM using llama.cpp"""
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,  # Context window size
        n_threads=4,  # CPU threads
        verbose=False
    )
    return llm

def summarize_text(llm: Llama, text: str, max_length: int = 500) -> str:
    """Generate a summary using the LLM"""
    prompt = f"""
    [INST] 1 .) Analyze the input text and generate 5 essential questions that, when answered, capture the main points and core meaning of the text.
    2.) When formulating your questions:
    > a. Address the central theme or argument
    > b. Identify key supporting ideas 
    > c. Highlight important facts or evidence 
    > d. Reveal the author's purpose or perspective
    > e. Explore any significant implications or conclusions. 
    3.) Answer all of your generated questions in a text with the conclusions. 
    Here is the text: 
    {text}
    [/INST]
    """
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_length,
        temperature=0.2,
    )
    return output["choices"][0]["message"]["content"]