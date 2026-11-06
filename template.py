"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # TODO: Import OpenAI, instantiate client, call chat.completions.create with parameters,
    #       measure execution start/end time, extract text and token usage, and return them.
    from openai import OpenAI

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Measure execution time
    start_time = time.perf_counter()

    # Call OpenAI Chat Completions API
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # Calculate latency
    latency = time.perf_counter() - start_time

    # Extract response text
    response_text = response.choices[0].message.content or ""

    # Extract token usage
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    # TODO: Initialize Gemini client, set config parameters, call generate_content,
    #       measure latency, extract response text and usage metadata, and return the tuple.
  
    # Import Google GenAI SDK
    from google import genai
    from google.genai import types

    # Initialize Gemini client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Configure generation parameters
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    # Measure execution time
    start_time = time.perf_counter()

    # Call Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    # Calculate latency
    latency = time.perf_counter() - start_time

    # Extract response text
    response_text = response.text or ""

    # Extract token usage statistics
    metadata = response.usage_metadata

    usage = {
        "input_tokens": metadata.prompt_token_count if metadata else 0,
        "output_tokens": metadata.candidates_token_count if metadata else 0
    }

    # Return results
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    # TODO: Initialize Anthropic client, create message, measure latency,
    #       extract content text and usage statistics, and return the tuple.
  
    # Import Anthropic SDK
    import anthropic

    # Initialize Anthropic client
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Measure execution time
    start_time = time.perf_counter()

    # Call Anthropic Messages API
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Calculate latency
    latency = time.perf_counter() - start_time

    # Extract response text
    response_text = "".join(
        block.text
        for block in response.content
        if block.type == "text"
    )

    # Extract token usage statistics
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    # Return results
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # TODO: Call call_openai with default gpt-4o model
    # TODO: Call call_openai with gpt-4o-mini model
    # TODO: Call call_gemini with default gemini-2.5-flash model
    # TODO: Calculate costs exactly based on input and output token counts using PRICING_1M_TOKENS
    #       Formula: Cost = (input_tokens * input_rate_per_1M + output_tokens * output_rate_per_1M) / 1,000,000
    # TODO: Assemble and return the comparison dictionary.

    # Call OpenAI GPT-4o
    gpt4o_response, gpt4o_latency, gpt4o_usage = call_openai(
        prompt=prompt,
        model=OPENAI_MODEL
    )

    # Call OpenAI GPT-4o-mini
    mini_response, mini_latency, mini_usage = call_openai(
        prompt=prompt,
        model=OPENAI_MINI_MODEL
    )

    # Call Google Gemini 2.5 Flash
    gemini_response, gemini_latency, gemini_usage = call_gemini(
        prompt=prompt,
        model=GEMINI_MODEL
    )

    # Calculate exact USD cost for each model
    def calculate_cost(model_name: str, usage: dict) -> float:
        pricing = PRICING_1M_TOKENS[model_name]

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        cost = (
            input_tokens * pricing["input"]
            + output_tokens * pricing["output"]
        ) / 1_000_000

        return cost

    # Assemble comparison dictionary
    results = {
        "gpt4o": {
            "response": gpt4o_response,
            "latency": gpt4o_latency,
            "cost": calculate_cost("gpt-4o", gpt4o_usage),
            "input_tokens": gpt4o_usage["input_tokens"],
            "output_tokens": gpt4o_usage["output_tokens"]
        },

        "gpt4o_mini": {
            "response": mini_response,
            "latency": mini_latency,
            "cost": calculate_cost("gpt-4o-mini", mini_usage),
            "input_tokens": mini_usage["input_tokens"],
            "output_tokens": mini_usage["output_tokens"]
        },

        "gemini_flash": {
            "response": gemini_response,
            "latency": gemini_latency,
            "cost": calculate_cost("gemini-2.5-flash", gemini_usage),
            "input_tokens": gemini_usage["input_tokens"],
            "output_tokens": gemini_usage["output_tokens"]
        }
    }

    # Return comparison results
    return results



# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    # TODO: Setup interactive session, prompt user for input, stream response, and update history.

    # Import Google GenAI SDK
    from google import genai
    from google.genai import types

    # Get Gemini API Key
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found!")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Initialize conversation history
    history = []

    print("\n=== Gemini 2.5 Streaming Chatbot ===")
    print("Type 'quit' or 'exit' to stop the chatbot.\n")

    # Create chat session
    chat = client.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=512
        )
    )

    while True:

        # Get user input
        user_input = input("You: ").strip()

        # Exit chatbot
        if user_input.lower() in ["quit", "exit"]:
            print("Chatbot: Goodbye!")
            break

        # Skip empty input
        if not user_input:
            continue

        try:
            print("Gemini: ", end="", flush=True)

            response_text = ""

            # Stream Gemini response
            for chunk in chat.send_message_stream(user_input):

                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_text += chunk.text

            print("\n")

            # Update conversation history
            history.append({
                "user": user_input,
                "assistant": response_text
            })

            # Keep only the last 3 conversation turns
            history = history[-3:]

            # Rebuild chat session with recent history
            chat = client.chats.create(
                model=GEMINI_MODEL,
                history=[
                    item
                    for turn in history
                    for item in [
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=turn["user"])]
                        ),
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=turn["assistant"])]
                        )
                    ]
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    top_p=0.9,
                    max_output_tokens=512
                )
            )

        except Exception as error:
            print(f"\nError: {error}")



# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff

    # Retry function with exponential backoff
    last_exception = None

    for attempt in range(max_retries + 1):

        try:
            # Execute the function
            return fn()

        except Exception as error:

            last_exception = error

            # Check whether all retries have been exhausted
            if attempt == max_retries:
                raise

            # Calculate exponential backoff delay
            delay = base_delay * (2 ** attempt)

            # Wait before the next retry
            print(
                f"Attempt {attempt + 1} failed: {error}. "
                f"Retrying in {delay:.2f} seconds..."
            )

            time.sleep(delay)

    raise last_exception



# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, and inject the original "prompt".

    # Initialize results list
    results = []

    # Iterate through each prompt
    for prompt in prompts:

        # Compare models using the current prompt
        comparison = compare_models(prompt)

        # Add original prompt to comparison result
        comparison["prompt"] = prompt

        # Append result to results list
        results.append(comparison)

    # Return all comparison results
    return results



# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    # TODO: Build and return the formatted table string. Truncate response to 50 chars for clean display.

    # Define table headers
    headers = [
        "Prompt",
        "Model",
        "Response (truncated)",
        "Latency",
        "Tokens (In/Out)",
        "Cost (USD)"
    ]

    # Initialize table rows
    rows = []

    # Iterate through comparison results
    for result in results:

        # Get original prompt
        prompt = str(result.get("prompt", ""))

        # Define model names and their corresponding keys
        models = [
            ("gpt4o", "GPT-4o"),
            ("gpt4o_mini", "GPT-4o-mini"),
            ("gemini_flash", "Gemini 2.5 Flash")
        ]

        # Iterate through each model
        for model_key, model_name in models:

            if model_key not in result:
                continue

            stats = result[model_key]

            # Extract response and truncate to 50 characters
            response = str(stats["response"])

            if len(response) > 50:
                response = response[:47] + "..."

            # Extract latency
            latency = stats["latency"]

            # Extract token usage
            input_tokens = stats["input_tokens"]
            output_tokens = stats["output_tokens"]

            # Extract cost
            cost = stats["cost"]

            # Add row to table
            rows.append([
                prompt.replace("\n", " ").replace("|", "\\|"),
                model_name,
                response.replace("\n", " ").replace("|", "\\|"),
                f"{latency:.2f}s",
                f"{input_tokens}/{output_tokens}",
                f"${cost:.8f}"
            ])

    # Build Markdown table
    table = []

    # Add header row
    table.append("| " + " | ".join(headers) + " |")

    # Add separator row
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Add data rows
    for row in rows:
        table.append("| " + " | ".join(row) + " |")

    # Return formatted Markdown table
    return "\n".join(table)



# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
