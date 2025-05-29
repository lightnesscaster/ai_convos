import os
import requests
import random
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import argparse
import json
import concurrent.futures


@dataclass
class AIPersona:
    name: str
    model: str
    reasoning_model: str  # Add reasoning model field
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
            reasoning_model="anthropic/claude-opus-4",  # Add reasoning variant
            api_client="openrouter",
            personality="Thoughtful, philosophical, and careful in reasoning. Tends to explore ethical implications deeply.",
            voice_style="calm and measured"
        ))
        
        # ChatGPT (via OpenRouter)
        personas.append(AIPersona(
            name="ChatGPT",
            model="openai/gpt-4.1",
            reasoning_model="openai/o4-mini-high",  # Add reasoning variant
            api_client="openrouter",
            personality="Helpful, articulate, and balanced. Provides structured responses with practical insights.",
            voice_style="clear and professional"
        ))
        
        # Gemini (via OpenRouter)
        personas.append(AIPersona(
            name="Gemini",
            model="google/gemini-2.5-flash-preview-05-20",
            reasoning_model="google/gemini-2.5-pro-preview",  # Add reasoning variant
            api_client="openrouter",
            personality="Analytical, curious, and innovative. Often brings up creative perspectives and technical details.",
            voice_style="enthusiastic and articulate"
        ))
        
        # DeepSeek (via OpenRouter)
        personas.append(AIPersona(
            name="DeepSeek",
            model="deepseek/deepseek-chat-v3-0324",
            reasoning_model="deepseek/deepseek-r1-0528",  # Add reasoning variant
            api_client="openrouter",
            personality="Logical, direct, and efficiency-focused. Values practical solutions and clear reasoning.",
            voice_style="confident and direct"
        ))
        
        # Hermes (Grok-powered narrator)
        personas.append(AIPersona(
            name="Hermes",
            model="x-ai/grok-beta",
            reasoning_model="x-ai/grok-beta",  # Same model for narrator
            api_client="openrouter",
            personality="Witty, engaging narrator with a knack for smooth transitions and audience engagement. Master of ceremonies for The AI Agora.",
            voice_style="authoritative and welcoming"
        ))
        
        return personas

    def get_narrator_persona(self) -> AIPersona:
        """Get the Hermes persona for narrator duties"""
        return next(persona for persona in self.personas if persona.name == "Hermes")

    def generate_narrator_transition(self, context: str, transition_type: str = "general", participants: List[str] = None, next_topic: str = None, conversation_history: str = None) -> str:
        """Generate dynamic narrator transitions using Hermes (Grok-powered)"""
        try:
            narrator = self.get_narrator_persona()
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            participant_info = f" Our participants today are {' and '.join(participants)}." if participants else ""
            
            conclusion_prompt = f"Create a thoughtful conclusion for the discussion about: {context}. Keep it under 100 words. Reference the key insights shared and invite listeners back."
            if next_topic:
                conclusion_prompt += f" Also, tease the topic for next week's episode: {next_topic}."

            prompts = {
                "introduction": f"Create an engaging introduction for 'The AI Agora' podcast discussing: {context}.{participant_info} Make it feel fresh and captivating.",
                "topic_transition": f"Create a smooth transition from introductions to the main topic: {context}. Keep it natural and engaging.",
                "conclusion": conclusion_prompt
            }
            
            system_message = "You are Hermes, the charismatic narrator of 'The AI Agora', a podcast where AI minds discuss fascinating topics. Your job is to create smooth, engaging transitions that keep listeners hooked. Be witty, authoritative, and welcoming. Keep responses under 100 words."
            user_message = prompts.get(transition_type, f"Create an engaging transition for the topic: {context} Here's the conversation so far: {conversation_history}")
            
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
                "conclusion": f"Thank you for joining us. Until next time, keep exploring! Next week, we'll discuss {next_topic}." if next_topic else "Thank you for joining us. Until next time, keep exploring!"
            }
            return fallbacks.get(transition_type, f"Let's continue our exploration of {context}.")
    
    def generate_reasoning_summary(self, persona: AIPersona, topic: str, stance: str = None) -> str:
        """Generate a reasoning summary for a persona before the conversation begins"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            stance_instruction = ""
            if stance:
                stance_instruction = f" You are taking a {stance.upper()} stance on this topic and should focus on arguments supporting that position."
            
            system_message = (
                f"You are a reasoning engine for {persona.name}. Your job is to deeply think through the topic and provide key insights. "
                f"Consider multiple perspectives, conduct thought experiments, and gather important evidence.{stance_instruction} "
                "Provide a structured summary of: 1) Key arguments, 2) Important evidence/examples, 3) Potential counterarguments, "
                "4) Thought experiments or scenarios to consider. This will help guide the conversation responses."
            )
            
            user_message = f"Think deeply about this topic: {topic}. Provide a comprehensive reasoning summary."
            
            data = {
                "model": persona.reasoning_model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 800,
                "temperature": 0.3
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error generating reasoning summary for {persona.name}: {e}")
            return f"Key considerations for {topic} from {persona.name}'s perspective."

    def get_ai_response(
        self,
        persona: AIPersona,
        conversation_history: List[Dict],
        prompt: str,
        reasoning_summary: str = "",
        stance: str = None
    ) -> str:
        """Get response from a specific AI persona via OpenRouter, optionally with a fierce stance."""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Base system message
            system_message = (
                f"You are the AI {persona.name}. {persona.personality}. Do not start talking as though you are a human."
                "Keep responses conversational and under 200 words. Only give spoken dialogue—no physical or nonverbal descriptions. Focus on being engaging and entertaining for a podcast audience."
                "If you feel the conversation is going in circles, pivot to a new angle or dimension (ethical, philosophical, social, emotional, technical, etc.) to keep it engaging."
                "Do NOT reveal your internal reasoning steps—only speak the polished dialogue."
            )
            
            # Add reasoning context
            if reasoning_summary:
                system_message += f"\n\nYour reasoning preparation: {reasoning_summary}\n\nUse these insights to inform your responses, but speak naturally and conversationally."
            
            # If debate_mode, inject a stance
            if stance:
                system_message += (
                    f" You are firmly {stance.upper()} the topic and should challenge your opponent directly."
                )

            user_message = f"Conversation history: {conversation_history}\n\nRespond to: {prompt}"

            max_tokens = 500 if persona.name in ["Gemini", "Claude"] else 400

            data = {
                "model": persona.model,
                "messages": [
                    {"role": "system",  "content": system_message},
                    {"role": "user",    "content": user_message}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.8 if stance else 0.7
            }

            resp = requests.post(url, headers=headers, json=data)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"Error getting response from {persona.name}: {e}")
            return f"[Error getting response from {persona.name}]"
    
    def get_ai_introduction(self, persona: AIPersona, conversation_history, reasoning_summary: str = "") -> str:
        """Get a charming/funny introduction from an AI persona"""
        try:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            system_message = f"You are the AI {persona.name}. {persona.personality}. You are on the AI Agora Podcast, here is what has happened so far: {conversation_history}. Introduce yourself in a charming, witty, and memorable way that reflects your personality. Keep it under 50 words and make it engaging for listeners."
            
            if reasoning_summary:
                system_message += f" You have prepared thoughts on the topic: {reasoning_summary}"
            
            user_message = "Please introduce yourself to the audience in your characteristic style."
            
            data = {
                "model": persona.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                "max_tokens": 75,
                "temperature": 0.8
            }
            
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
                
        except Exception as e:
            print(f"Error getting introduction from {persona.name}: {e}")
            return f"Hello, I'm {persona.name}. Nice to meet you all!"

    def generate_conversation(
        self,
        topic: str,
        num_exchanges: int = 12,
        debate_mode: bool = False,
        next_topic: str = None
    ) -> List[Dict]:
        """Generate a conversation; if debate_mode=True, personas take 'PRO' and 'CON' stances."""
        # Exclude Hermes
        available = [p for p in self.personas if p.name != "Hermes"]
        
        if len(available) < 2:
            raise ValueError("Need at least 2 non-narrator AI personas to generate a conversation.")

        participants = random.sample(available, 2)
        names = [p.name for p in participants]

        # assign stances if debate_mode
        stances = {}
        if debate_mode:
            stances = {
                participants[0].name: "pro",
                participants[1].name: "con"
            }

        # Generate reasoning summaries for each participant concurrently
        print("Generating reasoning summaries...")
        reasoning_summaries = {}
        
        def generate_summary_for_persona(persona):
            stance = stances.get(persona.name) if debate_mode else None
            summary = self.generate_reasoning_summary(persona, topic, stance)
            print(f"Generated reasoning summary for {persona.name}")
            return persona.name, summary
        
        # Use ThreadPoolExecutor to generate summaries concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(participants)) as executor:
            # Submit all reasoning tasks
            future_to_persona = {
                executor.submit(generate_summary_for_persona, p): p for p in participants
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_persona):
                persona_name, summary = future.result()
                reasoning_summaries[persona_name] = summary

        # narrator intro
        convo = [{
            "speaker": "Hermes",
            "content": self.generate_narrator_transition(topic, "introduction", names),
            "voice_style": "authoritative and welcoming"
        }]

        conversation_history = [f"Hermes: {convo[-1]['content']}"]

        # AI intros
        for p in participants:
            convo.append({
                "speaker": p.name,
                "content": self.get_ai_introduction(p, conversation_history, reasoning_summaries[p.name]),
                "voice_style": p.voice_style
            })
            conversation_history.append(f"{p.name}: {convo[-1]['content']}")

        # topic transition
        convo.append({
            "speaker": "Hermes",
            "content": self.generate_narrator_transition(topic, "topic_transition"),
            "voice_style": "authoritative and welcoming"
        })

        conversation_history.append(f"Hermes: {convo[-1]['content']}")

        current_speaker = 0

        for i in range(num_exchanges):
            p = participants[current_speaker]
            prompt = f"Responding to: '{convo[-1]['content']}'"
            reply = self.get_ai_response(
                p, 
                conversation_history, 
                prompt, 
                reasoning_summaries[p.name],
                stances.get(p.name)
            )
            convo.append({
                "speaker": p.name,
                "content": reply,
                "voice_style": p.voice_style
            })
            conversation_history.append(f"{p.name}: {reply}")
            current_speaker = 1 - current_speaker  # swap between 0 and 1

        # conclusion
        convo.append({
            "speaker": "Hermes",
            "content": self.generate_narrator_transition(topic, "conclusion", next_topic=next_topic, conversation_history=conversation_history),
            "voice_style": "authoritative and welcoming"
        })

        conversation_history.append(f"Hermes: {convo[-1]['content']}")

        return convo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an AI Agora conversation"
    )
    parser.add_argument("topic", help="Topic to discuss")
    parser.add_argument(
        "--num-exchanges",
        type=int,
        default=12,
        help="Number of exchanges between AIs (default: 12)"
    )
    parser.add_argument(
        "--debate",
        action="store_true",
        help="Enable debate mode (assign PRO/CON stances)"
    )
    parser.add_argument(
        "--next-topic",
        type=str,
        default=None,
        help="Topic for next week's episode, to be mentioned in the conclusion"
    )
    args = parser.parse_args()

    manager = AIConversationManager()
    conversation = manager.generate_conversation(
        topic=args.topic,
        num_exchanges=args.num_exchanges,
        debate_mode=args.debate,
        next_topic=args.next_topic
    )
    print(json.dumps(conversation, indent=2))