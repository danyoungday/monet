"""
Base Agent class for the other agents to inherit from.
"""
import openai


class Agent():
    """
    A simple OpenAI agent.
    """
    def __init__(self, system_prompt: str, model: str, temperature: float, log_name: str = None):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

        self.history = []

        self.log_name = log_name
        if log_name:
            with open(f"log/{log_name}.txt", "a", encoding="utf-8") as f:
                f.write("--- New Session ---\n\n")
                f.write(f"System Prompt:\n{system_prompt}\n\n")

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

            # Logging
            if self.log_name:
                with open(f"log/{self.log_name}.txt", "a", encoding="utf-8") as f:
                    f.write(f"User:\n{content[0]['text']}\n\nResponse:\n{reply}\n\n")

            if save_history:
                self.history.append({"role": "user", "content": content})

            return reply

        except Exception as e:
            return f"Error generating response: {str(e)}"
