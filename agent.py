"""
Base Agent class for the other agents to inherit from.
"""
import openai


class Agent():
    """
    A simple OpenAI agent.
    """
    def __init__(self, system_prompt: str, model: str, temperature: float):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

        self.history = []

    def set_history(self, history: list[dict]):
        """
        Set the message history for the agent.
        """
        self.history = history

    def generate_response(self, content: list[dict] | str, save_history: bool = False) -> str:
        """
        Call the OpenAI API to generate a response based on the input content.
        """
        try:
            # Construct system prompt and message history
            messages = [{"role": "system", "content": self.system_prompt}]
            for message in self.history:
                messages.append(message)

            # Default convert string to list of dicts format
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            # Add our current user content to messages
            messages.append({"role": "user", "content": content})

            # Run it all through OpenAI
            response = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )
            reply = response.choices[0].message.content

            if save_history:
                self.history.append({"role": "user", "content": content})

            return reply

        except Exception as e:
            return f"Error generating response: {str(e)}"
