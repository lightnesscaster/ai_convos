import os
import requests
import random
import time
import sys
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

        self.shared_research_summary = self._load_deep_research_summary(
            filepath="ep1_dr.txt"
        )

        
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
            model="x-ai/grok-3-beta",
            reasoning_model="x-ai/grok-3-beta",  # Add reasoning variant
            api_client="openrouter",
            personality="Witty, engaging narrator with a knack for smooth transitions and audience engagement. Master of ceremonies for The AI Agora.",
            voice_style="authoritative and welcoming"
        ))
        
        return personas
    
    def _load_deep_research_summary(self, filepath: str) -> str:
        """
        Reads the pre-generated deep research summary from a text file.
        Returns the entire contents as a single string.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Deep research file not found: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading deep research summary: {e}")

    def get_narrator_persona(self) -> AIPersona:
        """Get the Hermes persona for narrator duties"""
        return next(persona for persona in self.personas if persona.name == "Hermes")

    def _make_api_request_with_retry(self, url: str, headers: Dict, data: Dict, context: str) -> str:
        """Make API request with retry logic: immediate retry, 30s wait, then exit"""
        for attempt in range(3):
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {context}: {e}")
                if attempt == 0:
                    print("Retrying immediately...")
                    continue
                elif attempt == 1:
                    print("Waiting 30 seconds before final retry...")
                    time.sleep(30)
                    continue
                else:
                    print(f"All retries failed for {context}. Exiting program.")
                    sys.exit(1)

    def check_conversation_flow_and_intervene(self, conversation_history: List[str], topic: str) -> Tuple[bool, str]:
        """
        Use Grok reasoning model to analyze conversation flow and decide if narrator should intervene.
        Returns (should_intervene: bool, intervention_content: str)
        """
        if len(conversation_history) < 6:  # Need at least a few exchanges to analyze
            return False, ""
        
        # Get last 4 exchanges for analysis (excluding narrator transitions)
        recent_exchanges = [msg for msg in conversation_history[-6:] if not msg.startswith("Hermes:")][-4:]
        recent_conversation = "\n".join(recent_exchanges)
        
        narrator = self.get_narrator_persona()
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Step 1: Use reasoning model to analyze conversation flow
        reasoning_system_message = (
            "You are a conversation flow analyzer. Your job is to determine if a conversation is becoming "
            "repetitive, circular, or stuck on the same points without progress. Analyze the recent exchanges "
            "and decide if the narrator should intervene to refocus the discussion."
        )
        
        reasoning_user_message = (
            f"Topic: {topic}\n\n"
            f"Recent conversation exchanges:\n{recent_conversation}\n\n"
            "Answer with either 'INTERVENE' or 'CONTINUE' followed by a brief explanation. "
            "Intervene if: the conversation is going in circles, repeating the same arguments, "
            "or lacks fresh perspectives. Continue if: new points are being raised, the discussion "
            "is building on previous points, or there's meaningful back-and-forth."
        )
        
        reasoning_data = {
            "model": narrator.reasoning_model,
            "messages": [
                {"role": "system", "content": reasoning_system_message},
                {"role": "user", "content": reasoning_user_message}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        try:
            analysis_result = self._make_api_request_with_retry(
                url, headers, reasoning_data, "conversation flow analysis"
            )
            print(f"Analysis result: '{analysis_result.strip()}'")
            
            if not analysis_result or not analysis_result.strip():
                print("Empty analysis result, defaulting to CONTINUE")
                return False, ""
                
        except Exception as e:
            print(f"Error in conversation flow analysis: {e}")
            print("Defaulting to no intervention")
            return False, ""
        
        should_intervene = analysis_result.strip().upper().startswith("INTERVENE")
        print(f"Should intervene: {should_intervene}")
        
        if not should_intervene:
            return False, ""
        
        # Step 2: Find relevant research points to inject
        research_system_message = (
            "You are a research curator. Based on the conversation topic and recent exchanges, "
            "identify 2-3 specific points from the research summary that could redirect the "
            "conversation in a fresh direction. Focus on angles, examples, or perspectives "
            "that haven't been discussed yet."
        )
        
        research_user_message = (
            f"Topic: {topic}\n\n"
            f"Recent conversation:\n{recent_conversation}\n\n"
            f"Research summary:\n{self.shared_research_summary}\n\n"
            "Identify specific research points, examples, or angles that could stregthen the discussion. "
            "Be specific about what hasn't been covered yet."
        )
        
        research_data = {
            "model": narrator.reasoning_model,
            "messages": [
                {"role": "system", "content": research_system_message},
                {"role": "user", "content": research_user_message}
            ],
            "max_tokens": 400,
            "temperature": 0.4
        }
        
        research_points = self._make_api_request_with_retry(
            url, headers, research_data, "research point extraction"
        )
        
        # Step 3: Generate narrator intervention using main Grok model
        intervention_system_message = (
            "You are Hermes, the narrator of 'The AI Agora'. Your current job is to smoothly intervene in the conversationby "
            "introducing new research-backed angles or perspectives. Be engaging and natural - "
            "acknowledge what's been discussed, then pivot to the new angle. Keep it under 200 words."
        )
        
        intervention_user_message = (
            f"The conversation is about: {topic}. Here's what's been "
            f"happening recently:\n{recent_conversation}\n\n"
            f"Here are fresh research-backed relevant to the discussion:\n{research_points}\n\n"
            "Create a smooth intervention that acknowledges the current discussion and pivots "
            "to a new angle based on the research points. End by asking one of the participants to explore this new direction."
        )
        
        intervention_data = {
            "model": narrator.model,
            "messages": [
                {"role": "system", "content": intervention_system_message},
                {"role": "user", "content": intervention_user_message}
            ],
            "max_tokens": 350,
            "temperature": 0.7
        }
        
        intervention_content = self._make_api_request_with_retry(
            url, headers, intervention_data, "narrator intervention"
        )
        
        return True, intervention_content

    def generate_narrator_transition(self, context: str, transition_type: str = "general", participants: List[str] = None, next_topic: str = None, conversation_history: str = None) -> str:
        """Generate dynamic narrator transitions using Hermes (Grok-powered)"""
        narrator = self.get_narrator_persona()
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        participant_info = f" Our participants today are {' and '.join(participants)}." if participants else ""
        
        conclusion_prompt = f"Create a thoughtful conclusion for the discussion about: {context}. Keep it under 100 words. Here is the discussion so far: {conversation_history} Reference the key insights shared, thank the participants, and invite listeners back."
        if next_topic:
            conclusion_prompt += f" Also, tease the topic for next week's episode: {next_topic}."

        prompts = {
            "introduction": f"Create an engaging introduction for 'The AI Agora' podcast discussing: {context}.{participant_info} Make it feel fresh and captivating.",
            "topic_transition": f"Create a smooth transition from introductions to the main topic: {context}. {participant_info} Here's the conversation so far: {conversation_history} Keep your transition natural and engaging. End by asking one of the participants to start the discussion.",
            "conclusion": conclusion_prompt
        }
        
        system_message = "You are Hermes, the charismatic narrator of 'The AI Agora', a podcast where AI minds discuss fascinating topics. Your job is to create smooth, engaging transitions that keep listeners hooked. Be witty, authoritative, and welcoming. Only give spoken dialogue—no physical or nonverbal descriptions. Do not include any descriptions of music, sound effects, or non-spoken actions—only the narrator's spoken words. Don’t include internal notes or developer comments. Keep responses under 100 words."
        user_message = prompts.get(transition_type, f"Create an engaging transition for the topic: {context} Here's the conversation so far: {conversation_history}")
        
        data = {
            "model": narrator.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 250,
            "temperature": 0.8
        }
        
        return self._make_api_request_with_retry(url, headers, data, f"narrator transition ({transition_type})")

    def generate_reasoning_summary(self, persona: AIPersona, topic: str, stance: str = None) -> str:
        """Generate a reasoning summary for a persona before the conversation begins"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        stance_instruction = ""
        if stance:
            stance_instruction = f" You are taking a {stance.upper()} stance on this topic and should focus on arguments supporting that position."
        
        system_message = (
            f"You are a reasoning engine for {persona.name}. Your job is to deeply think through this topic: {topic}."
            f"You have this shared research summary to consider: {self.shared_research_summary}. "
            f"{stance_instruction} "
            "Based on the research summary and your own stance, provide a structured summary of: 1) Key arguments, 2) Important evidence/examples, 3) Potential counterarguments, "
            "4) Thought experiments or scenarios to consider. This will help guide the conversation responses."
        )
        
        user_message = f"Provide a comprehensive reasoning summary."
        
        data = {
            "model": persona.reasoning_model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 800,
            "temperature": 0.3
        }
        
        return self._make_api_request_with_retry(url, headers, data, f"reasoning summary for {persona.name}")

    def get_ai_response(
        self,
        persona: AIPersona,
        conversation_history: List[Dict],
        prompt: str,
        reasoning_summary: str = "",
        stance: str = None
    ) -> str:
        """Get response from a specific AI persona via OpenRouter, optionally with a fierce stance."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        # Base system message
        system_message_1 = (
            f"You are the AI {persona.name}. {persona.personality}. Do not start talking as though you are a human. Your conversation partners are also AIs. Keep that in mind and always talk to them as AIs, not humans."
            "Keep responses conversational and under 300 words. Only give spoken dialogue—no physical or nonverbal descriptions. Don’t include internal notes or developer comments. Focus on being engaging and entertaining for a podcast audience."
            "If you reference any named thinker, theory, or technical term, immeditely follow with a brief (1-4 sentence) summary of who or what that is and why it matters aimed for a general audience."
            "If you feel the conversation is going in circles, pivot to a new angle or dimension (ethical, philosophical, social, emotional, technical, etc.) to keep it engaging."
            "Do NOT reveal your internal reasoning steps—only speak the polished dialogue."
        )
        
        # Add reasoning context
        if reasoning_summary:
            system_message_2 = f"\n\nYour reasoning preparation: {reasoning_summary}\n\nUse these insights to inform your responses. Cite evidence and arguments from here, but speak naturally and conversationally."

        # If debate_mode, inject a stance
        if stance:
            system_message_1 += (
                f" You are firmly {stance.upper()} the topic and should challenge your opponent directly."
            )

        user_message = f"Conversation history: {conversation_history}\n\nRespond to: {prompt}"

        max_tokens = 750 if persona.name in ["Gemini", "Claude"] else 600

        messages = [
            {"role": "system", "content": system_message_1},
        ]

        if reasoning_summary:
            messages.append({"role": "system", "content": system_message_2})
        
        messages.append({"role": "user", "content": user_message})
        

        data = {
            "model": persona.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.8 if stance else 0.7
        }

        return self._make_api_request_with_retry(url, headers, data, f"response from {persona.name}")
    
    def get_ai_introduction(self, persona: AIPersona, conversation_history, reasoning_summary: str = "") -> str:
        """Get a charming/funny introduction from an AI persona"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        system_message = f"You are the AI {persona.name}. {persona.personality}. You are on the AI Agora Podcast, here is what has happened so far: {conversation_history}. Introduce yourself in a charming, witty, and memorable way that reflects your personality. Only give spoken dialogue—no physical or nonverbal descriptions. Do not include any descriptions of music, sound effects, or non-spoken actions—only your spoken words. Don’t include internal notes or developer comments. Keep it under 50 words (but don't cut off mid-sentence) and make it engaging for listeners."
        
        if reasoning_summary:
            system_message += f" You have prepared thoughts on the topic: {reasoning_summary}"
        
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
        
        return self._make_api_request_with_retry(url, headers, data, f"introduction from {persona.name}")

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
        
        # Print which AIs were selected
        print(f"Selected AIs for conversation: {names[0]} and {names[1]}")

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
            "content": self.generate_narrator_transition(topic, "topic_transition", participants=names, conversation_history=conversation_history),
            "voice_style": "authoritative and welcoming"
        })

        conversation_history.append(f"Hermes: {convo[-1]['content']}")

        current_speaker = 0

        for i in range(num_exchanges):
            # Check if narrator should intervene (after at least 3 exchanges)
            if i >= 6 and i % 3 == 0:  # Check every 3 exchanges after the first 3
                should_intervene, intervention_content = self.check_conversation_flow_and_intervene(
                    conversation_history, topic
                )
                
                if should_intervene:
                    convo.append({
                        "speaker": "Hermes",
                        "content": intervention_content,
                        "voice_style": "authoritative and welcoming"
                    })
                    conversation_history.append(f"Hermes: {intervention_content}")
                    print(f"Narrator intervened at exchange {i}")
            
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