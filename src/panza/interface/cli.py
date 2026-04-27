from panza.entities.instruction import EmailInstruction, SnippetInstruction, Instruction
from panza.writer import PanzaWriter


class PanzaCLI:
    def __init__(self, writer: PanzaWriter, **kwargs):
        self.writer = writer
        while True:
            user_input = input("Enter a command: ")
            if user_input == "exit":
                break
            else:
                instruction: Instruction = SnippetInstruction(user_input, context="")
                stream = self.writer.run(instruction, stream=False)
                for block in stream:
                    print(block, end="")
                print("\n")
