import os
import re

import openai


class Agent:
    """
    A simple agent with no history.
    """
    def __init__(self, system_prompt: str, model: str, temperature: float):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature

    def generate_response(self, prompt: str) -> str:
        """
        Call the OpenAI API to generate a response based on the prompt.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.append({"role": "user", "content": prompt})

            response = openai.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )
            reply = response.choices[0].message.content
            return reply
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "Error generating response."


class CodingAgent():
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def generate_and_save(self, sysprompt: str, prompt: str) -> str:
        coding_agent = Agent(sysprompt, model=self.model, temperature=self.temperature)
        code_output = coding_agent.generate_response(prompt)
        n_files = len([f for f in os.listdir("storage") if f.endswith(".py")])
        with open(f"storage/bike-{n_files}.py", "w", encoding="utf-8") as f:
            f.write(code_output)
        return code_output


class NoveltyAgent(Agent):
    def __init__(self, model: str, temperature: float):
        with open("sysprompts/novelty_evaluator.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
        super().__init__(system_prompt, model, temperature)

    def evaluate(self, code: str) -> int:
        prompt = "Evaluate the novelty of the following code:\n" + code
        output = self.generate_response(prompt)
        score = re.search(r"\{(\d+)\}", output)
        if score:
            return output, int(score.group(1))
        else:
            return output, -1  # Indicate that parsing failed


def main():
    with open("sysprompts/prompt_maker.txt", "r", encoding="utf-8") as f:
        prompt_maker_system_prompt = f.read().strip()
    prompt_maker_agent = Agent(prompt_maker_system_prompt, model="gpt-5-mini", temperature=1.0)

    user_prompt = "Prompt the coding agent to draw a picture of a bicycle."
    coding_agent_system_prompt = prompt_maker_agent.generate_response(user_prompt)

    print(coding_agent_system_prompt)

    coding_agent = CodingAgent(model="gpt-5-mini", temperature=1.0)
    code = coding_agent.generate_and_save(coding_agent_system_prompt, "Draw a bicycle")

    novelty_agent = NoveltyAgent(model="gpt-5-mini", temperature=1.0)
    novelty_output, score = novelty_agent.evaluate(code)
    print("Novelty evaluation output:", novelty_output)
    print("Novelty score:", score)


if __name__ == "__main__":
    main()
