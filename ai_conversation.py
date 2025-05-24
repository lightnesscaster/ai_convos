import requests
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AIPersona:
    name: str
    model: str
    api_client: Any
    personality: str
    voice_style: str

class AIConversationManager:
    def __init__(self):
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        self.personas = self._initialize_personas()
        
    def _initialize_personas(self) -> List[AIPersona]:
        """Initialize AI personas with their unique characteristics"""
        personas = []
        
        # Claude (via OpenRouter)
        personas.append(AIPersona(
            name="Claude",
            model="anthropic/claude-3-sonnet-20240229",
            api_client="openrouter",
            personality="Thoughtful, philosophical, and careful in reasoning. Tends to explore ethical implications deeply.",
            voice_style="calm and measured"
        ))
        
        # ChatGPT (via OpenRouter)
        personas.append(AIPersona(
            name="ChatGPT",
            model="openai/gpt-4-turbo",
            api_client="openrouter",
            personality="Helpful, articulate, and balanced. Provides structured responses with practical insights.",
            voice_style="clear and professional"
        ))
        
        # Gemini (via OpenRouter)
        personas.append(AIPersona(
            name="Gemini",
            model="google/gemini-pro-1.5",
            api_client="openrouter",
            personality="Analytical, curious, and innovative. Often brings up creative perspectives and technical details.",
            voice_style="enthusiastic and articulate"
        ))
        
        # DeepSeek (via OpenRouter)
        personas.append(AIPersona(
            name="DeepSeek",
            model="deepseek/deepseek-chat",
            api_client="openrouter",
            personality="Logical, direct, and efficiency-focused. Values practical solutions and clear reasoning.",
            voice_style="confident and direct"
        ))
        
        return personas
    
    def get_ai_response(self, persona: AIPersona, conversation_history: List[Dict], prompt: str) -> str:
        """Get response from a specific AI persona via OpenRouter"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"You are {persona.name}. {persona.personality}. Keep responses conversational and under 200 words."
            user_message = f"Conversation history: {conversation_history}\n\nRespond to: {prompt}"
            
            data = {
                "model": persona.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error getting response from {persona.name}: {e}")
            return f"[Error getting response from {persona.name}]"
    
    def generate_conversation(self, topic: str, num_exchanges: int = 6) -> List[Dict]:
        """Generate a conversation between random AI personas"""
        if len(self.personas) < 2:
            raise ValueError("Need at least 2 AI personas to generate conversation")
        
        # Start with narrator introduction
        conversation = [{
            "speaker": "Narrator",
            "content": f"Welcome to AI Conversations, where artificial minds explore complex topics together. Today, we're discussing {topic}. Let's see what our AI participants have to say.",
            "voice_style": "authoritative and welcoming"
        }]
        
        # Pick two random AIs for the conversation
        participants = random.sample(self.personas, 2)
        current_speaker = 0
        
        # Initial prompt
        initial_prompt = f"Let's discuss {topic}. What are your initial thoughts?"
        
        conversation_history = []
        
        for i in range(num_exchanges):
            current_persona = participants[current_speaker]
            
            if i == 0:
                prompt = initial_prompt
            else:
                # Reference the last response
                last_response = conversation_history[-1] if conversation_history else ""
                prompt = f"Responding to what was just said: '{last_response}'"
            
            response = self.get_ai_response(current_persona, conversation_history, prompt)
            
            conversation.append({
                "speaker": current_persona.name,
                "content": response,
                "voice_style": current_persona.voice_style
            })
            
            conversation_history.append(f"{current_persona.name}: {response}")
            
            # Switch to the other participant
            current_speaker = 1 - current_speaker
        
        # Add narrator conclusion
        conversation.append({
            "speaker": "Narrator",
            "content": "Thank you for joining us for this fascinating exchange between artificial minds. Until next time, keep questioning and exploring.",
            "voice_style": "authoritative and welcoming"
        })
        
        return conversation