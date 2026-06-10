import os
import sys

try:
    from pydantic import BaseModel, Field, ValidationError
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: Required packages (google-genai, pydantic) are not installed.")
    sys.exit(1)

# Import the environment function
from sandbox_env import run_environment

class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)

    try:
        client = genai.Client()
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        sys.exit(1)

    # Deliberately bad starting kappa value
    kappa = 1.0  
    
    for turn in range(1, 6):
        print(f"\n========== Turn {turn} ==========")
        
        # 1. Run the sandbox environment
        env_result = run_environment(kappa)
        print(f"Diagnostic Summary: {env_result['diagnostic_summary']}")
        print(f"Temperature Log: {env_result['temperature_log']}")
        
        # 2. Prepare the prompt for Gemini
        prompt = f"""
You are an AI controller managing a thermal dampener.
Current Kappa: {env_result['current_kappa']}
Temperature Log: {env_result['temperature_log']}
Final Temperature: {env_result['final_temperature']}
System State: {env_result['system_state']}

Your goal is to reach the PERFECT state (Kappa between 4.5 and 5.5).
Analyze the current situation and decide how to adjust Kappa to reach the optimal zone.
"""
        
        try:
            # 3. Call Gemini enforcing JSON response matching Pydantic schema
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ControlDecision,
                )
            )
            
            raw_json = response.text
            print("\n--- Raw JSON received from Gemini ---")
            print(raw_json)
            print("-------------------------------------\n")
            
            # 4. Parse the JSON response programmatically using Pydantic
            decision = ControlDecision.model_validate_json(raw_json)
            
            # 6. Apply the decision
            if decision.adjustment_action == "INCREASE":
                kappa += decision.delta_value
            elif decision.adjustment_action == "DECREASE":
                kappa -= decision.delta_value
            elif decision.adjustment_action == "HOLD":
                pass
            
            # 7. Print the updated kappa
            print(f"Action taken: {decision.adjustment_action} Kappa by {decision.delta_value}")
            print(f"Updated Kappa: {kappa}")
            
            if decision.system_state == "PERFECT":
                print("\nAI correctly identified PERFECT state!")
                # Optional: break if we want, or keep simulating. Let's keep simulating for 5 turns
                # as requested ("Run a closed loop for 5 turns").
                
        except ValidationError as e:
            print(f"JSON validation error (Invalid response from Gemini): {e}")
            sys.exit(1)
        except Exception as e:
            print(f"API call failed: {e}")
            sys.exit(1)

    print("\n--- Final Simulation Result ---")
    final_env = run_environment(kappa)
    print(f"Final State: {final_env['system_state']}, Kappa: {kappa}")

if __name__ == "__main__":
    main()
