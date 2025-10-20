"""
Base Agent class for the other agents to inherit from.
"""
import openai


class Agent:
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

    def set_history(self, history: list[list[dict]]):
        """
        Set the message history for the agent.
        """
        self.history = []
        for message_list in history:
            for message in message_list:
                self.history.append(message)

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
            print(f"Error calling OpenAI API: {e}")
            return "Error generating response."

    def generate_with_examples(self, prompt: str, examples: list[str]) -> str:
        """
        Generate using some few-shot examples.
        """
        full_prompt = prompt
        if len(examples) > 0:
            example_text = "\n\n".join(examples)
            full_prompt += f"\n\nThe following are some previously generated examples:\n\n{example_text}"
        return self.generate_response(full_prompt)
