import os
import sys

try:
    from google import genai
except ImportError:
    print("Error: 'google-genai' package is not installed. Please install it to run this script.")
    sys.exit(1)

def main():
    # Check if the API key is present in the environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)

    try:
        # The client automatically picks up GEMINI_API_KEY from the environment
        client = genai.Client()
        
        model_name = "gemini-2.5-flash"
        prompt = "Explain the difference between a stateful NumPy random generation process and a stateless JAX PRNG split operation in exactly one highly sarcastic sentence."
        
        # Call the Gemini model
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        
        print(f"Model: {model_name}")
        print("Response:")
        print(response.text)
        
    except Exception as e:
        print(f"API call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
