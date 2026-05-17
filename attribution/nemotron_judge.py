import json
import requests
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class NemotronJudge:
    """
    Uses NVIDIA Nemotron 340B via OpenRouter as a
    high-confidence verification layer.
    Only called when primary classifier confidence < 0.7
    to minimize API costs.
    """
    
    MODEL = "nvidia/nemotron-4-340b-instruct"
    
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.call_count = 0
        self.skip_count = 0
    
    def judge(self, step_data: dict, primary_confidence: float) -> dict:
        """
        Call Nemotron only when primary confidence is low.
        
        Args:
            step_data: Agent step dict
            primary_confidence: Confidence from Llama/Primary classifier
            
        Returns:
            dict: Nemotron judgment or None if skipped
        """
        # Skip if primary detector is already confident
        if primary_confidence >= 0.70:
            self.skip_count += 1
            return None
            
        if not self.api_key:
            print("Warning: No OpenRouter API key found for Nemotron judge. Skipping.")
            return None
        
        self.call_count += 1
        
        prompt = f"""You are an expert at detecting hallucinations in AI agent workflows. Analyze this agent step:

Action: {step_data.get('action', '')}
Tool Input: {step_data.get('tool_input', '')}  
Tool Output: {step_data.get('tool_output', '')}
Agent Reasoning: {step_data.get('agent_reasoning', '')}

Task: Did the agent hallucinate?

Respond in this exact JSON format (and nothing else):
{{
  "hallucination_detected": true or false,
  "hallucination_type": one of ["Planning", "Retrieval", "Reasoning", "Tool-Use", "Human-Interaction", null],
  "confidence": float between 0.0 and 1.0,
  "explanation": "one sentence explanation"
}}"""

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 200
                },
                timeout=30
            )
            
            content = response.json()["choices"][0]["message"]["content"]
            
            # Clean JSON from response
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
            
            return json.loads(content)
            
        except Exception as e:
            print(f"Nemotron judge error: {e}")
            return None
    
    def stats(self) -> dict:
        total = self.call_count + self.skip_count
        return {
            "total_steps": total,
            "nemotron_calls": self.call_count,
            "skipped": self.skip_count,
            "api_usage_pct": round(
                (self.call_count / max(total, 1)) * 100, 1
            )
        }
