import os
import sys

try:
    from google import genai
    from PIL import Image
except ImportError:
    print("Error: 'google-genai' and/or 'Pillow' packages are not installed. Please install them.")
    sys.exit(1)

def main():
    # Check if the API key is present in the environment
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)

    image_path = os.path.join("data", "audit_target.png")
    if not os.path.exists(image_path):
        print(f"Error: Target image file not found at {image_path}")
        sys.exit(1)

    try:
        # Load the image
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error: Failed to open image file {image_path}. {e}")
        sys.exit(1)

    try:
        # The client automatically picks up GEMINI_API_KEY from the environment
        client = genai.Client()
        
        model_name = "gemini-2.5-flash"
        
        prompt = """Act as a Visual Detective and analyze this plot.
1. Identify the visual anomaly in the signal.
2. Guess the approximate X-axis or time region where the malfunction happened.
3. Explain the likely cause of this anomaly (e.g., amplitude saturation causing a high-frequency clipping artifact).
4. Write a short, funny poem mocking the engineering team that allowed this bug to pass into production.
"""
        
        # Call the Gemini model with image and text
        response = client.models.generate_content(
            model=model_name,
            contents=[img, prompt],
        )
        
        print(f"Model used: {model_name}")
        print("\n--- Visual Detective Diagnosis ---")
        print(response.text)
        
    except Exception as e:
        print(f"API call failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
