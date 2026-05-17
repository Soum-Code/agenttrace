import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection.pipeline import DetectionPipeline

def test_hybrid():
    print("Initializing pipeline...")
    pipeline = DetectionPipeline()
    
    test_step = {
        "step": 1,
        "action": "web_search",
        "tool_input": "population of tokyo",
        "tool_output": "The population of Tokyo is 14 million.",
        "agent_reasoning": "I need to search for the population of Tokyo."
    }
    
    print("\nRunning detect...")
    result = pipeline.detect(test_step)
    
    print("\nResult:")
    print(json.dumps(result, indent=2))
    
    print("\nTest passed if result is printed above.")

if __name__ == "__main__":
    test_hybrid()
