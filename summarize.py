import re
from vllm_gen import get_completion_text

summary_prompt_file = "generation_prompts/summary_prompt.txt"

# Explantion, answer grammar
e_a_grammar = """root ::= " " line "\nEvaluation: " answer

# Yes or no answer
answer ::= "Yes" | "No"

# String
line ::= [^\r\n\x0b\x0c\x85\u2028\u2029|:]+
"""

e_a_regex = " [^\r\n\x0b\x0c\x85\u2028\u2029|:]+\nEvaluation: (Yes|No)"

# Explanation, relevance score grammar
e_r_grammar = """<root> ::= " " <line> "\nRelevance score: " <score>

# String
<line> ::= [^\r\n\x0b\x0c\x85\u2028\u2029|:]+

# Score
<score> ::= ("0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10")
"""

e_r_regex = " [^\r\n\x0b\x0c\x85\u2028\u2029|:]+\nRelevance score: (10|[0-9])"

# Function to check if a set of notes is useful, returns the explanation and the relevance score
def pass_test_usefulness(question, notes):
    # Read the prompt from the file
    with open(summary_prompt_file, "r") as f:
        prompt = f.read()