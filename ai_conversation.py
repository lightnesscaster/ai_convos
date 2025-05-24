import requests
import random
import time
from typing import List, Dict, Any
from dataclasses import dataclass
import os


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
            model="anthropic/claude-sonnet-4",
            api_client="openrouter",
            personality="Thoughtful, philosophical, and careful in reasoning. Tends to explore ethical implications deeply.",
            voice_style="calm and measured"
        ))
        
        # ChatGPT (via OpenRouter)
        personas.append(AIPersona(
            name="ChatGPT",
            model="openai/gpt-4.1",
            api_client="openrouter",
            personality="Helpful, articulate, and balanced. Provides structured responses with practical insights.",
            voice_style="clear and professional"
        ))
        
        # Gemini (via OpenRouter)
        personas.append(AIPersona(
            name="Gemini",
            model="google/gemini-2.5-flash-preview-05-20",
            api_client="openrouter",
            personality="Analytical, curious, and innovative. Often brings up creative perspectives and technical details.",
            voice_style="enthusiastic and articulate"
        ))
        
        # DeepSeek (via OpenRouter)
        personas.append(AIPersona(
            name="DeepSeek",
            model="deepseek/deepseek-chat-v3-0324",
            api_client="openrouter",
            personality="Logical, direct, and efficiency-focused. Values practical solutions and clear reasoning.",
            voice_style="confident and direct"
        ))
        
        # Hermes (Grok-powered narrator)
        personas.append(AIPersona(
            name="Hermes",
            model="x-ai/grok-beta",
            api_client="openrouter",
            personality="Witty, engaging narrator with a knack for smooth transitions and audience engagement. Master of ceremonies for The AI Agora.",
            voice_style="authoritative and welcoming"
        ))
        
        return personas

    def get_narrator_persona(self) -> AIPersona:
        """Get the Hermes persona for narrator duties"""
        return next(persona for persona in self.personas if persona.name == "Hermes")

    def generate_narrator_transition(self, context: str, transition_type: str = "general", participants: List[str] = None) -> str:
        """Generate dynamic narrator transitions using Hermes (Grok-powered)"""
        try:
            narrator = self.get_narrator_persona()
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            participant_info = f" Our participants today are {' and '.join(participants)}." if participants else ""
            
            prompts = {
                "introduction": f"Create an engaging introduction for 'The AI Agora' podcast discussing: {context}.{participant_info} Make it feel fresh and captivating.",
                "topic_transition": f"Create a smooth transition from introductions to the main topic: {context}. Keep it natural and engaging.",
                "conclusion": f"Create a thoughtful conclusion for the discussion about: {context}. Reference the key insights shared and invite listeners back."
            }
            
            system_message = "You are Hermes, the charismatic narrator of 'The AI Agora', a podcast where AI minds discuss fascinating topics. Your job is to create smooth, engaging transitions that keep listeners hooked. Be witty, authoritative, and welcoming. Keep responses under 100 words."
            user_message = prompts.get(transition_type, f"Create an engaging transition for the topic: {context}")
            
            data = {
                "model": narrator.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 150,
                "temperature": 0.8
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error generating narrator transition: {e}")
            # Fallback to static content
            fallbacks = {
                "introduction": f"Welcome to The AI Agora, where artificial minds explore {context}.",
                "topic_transition": f"Now let's dive into our discussion about {context}.",
                "conclusion": "Thank you for joining us. Until next time, keep exploring!"
            }
            return fallbacks.get(transition_type, f"Let's continue our exploration of {context}.")
    
    def get_ai_response(self, persona: AIPersona, conversation_history: List[Dict], prompt: str) -> str:
        """Get response from a specific AI persona via OpenRouter"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"You are {persona.name}. {persona.personality}. Keep responses conversational and under 200 words. Only provide spoken dialogue - do not include any non-verbal cues, gestures, actions, stage directions, or descriptions of body language. This is a podcast conversation."
            user_message = f"Conversation history: {conversation_history}\n\nRespond to: {prompt}"
            
            # Increase max_tokens to prevent responses from getting cut off
            # Gemini and other analytical AIs need more space for complete thoughts
            max_tokens = 500 if persona.name in ["Gemini", "Claude"] else 400
            
            data = {
                "model": persona.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error getting response from {persona.name}: {e}")
            return f"[Error getting response from {persona.name}]"
    
    def get_ai_introduction(self, persona: AIPersona) -> str:
        """Get a charming/funny introduction from an AI persona"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"You are {persona.name}. {persona.personality}. Introduce yourself in a charming, witty, and memorable way that reflects your personality. Keep it under 100 words and make it engaging for listeners."
            user_message = "Please introduce yourself to the audience in your characteristic style."
            
            data = {
                "model": persona.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 150,
                "temperature": 0.8
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error getting introduction from {persona.name}: {e}")
            return f"Hello, I'm {persona.name}. Nice to meet you all!"

    def generate_conversation(self, topic: str, num_exchanges: int = 6) -> List[Dict]:
        """Generate a conversation between random AI personas"""
        if len(self.personas) < 2:
            raise ValueError("Need at least 2 AI personas to generate conversation")
        
        # Pick two random AIs for the conversation (excluding Hermes who is always the narrator/host)
        available_personas = [p for p in self.personas if p.name != "Hermes"]
        participants = random.sample(available_personas, 2)
        participant_names = [p.name for p in participants]
        
        # Start with dynamic narrator introduction (Hermes as host)
        intro_content = self.generate_narrator_transition(topic, "introduction", participant_names)
        conversation = [{
            "speaker": "Hermes",
            "content": intro_content,
            "voice_style": "authoritative and welcoming"
        }]
        
        # AI introductions
        for persona in participants:
            introduction = self.get_ai_introduction(persona)
            conversation.append({
                "speaker": persona.name,
                "content": introduction,
                "voice_style": persona.voice_style
            })
        
        # Dynamic narrator transition to main topic (Hermes as host)
        transition_content = self.generate_narrator_transition(topic, "topic_transition")
        conversation.append({
            "speaker": "Hermes",
            "content": transition_content,
            "voice_style": "authoritative and welcoming"
        })
        
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
        
        # Add dynamic narrator conclusion (Hermes as host)
        conclusion_content = self.generate_narrator_transition(topic, "conclusion")
        conversation.append({
            "speaker": "Hermes",
            "content": conclusion_content,
            "voice_style": "authoritative and welcoming"
        })
        
        return conversation