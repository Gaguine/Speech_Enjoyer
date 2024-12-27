"""
This should be a straightforward project.

1. The text should be already cleaned, yet to be parsed.
2. create an inferance, where the LLM model analyses each phrase.
3. Store the gained information(probably a semantic value)
4. Plot the data to see the emotion dynamics, using matplot lib
"""


import torch

print(torch.cuda.is_available())

print(torch.__version__)  # Prints the PyTorch version
print(torch.version.cuda)


