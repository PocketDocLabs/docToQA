import re
from transformers import AutoTokenizer
from vllm_gen import get_completion_text

in_context_prompt_file = "./rubric_prompts/out-of-context-llama3.txt"

situ_aprop_prompt_file = "./rubric_prompts/situationally-appropriate-llama3.txt"

usefulness_prompt_file = "./context_prompts/usefulness-check-llama3.txt"

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


# Function to check if a question is in context, returns True if the question is in context, False otherwise and the explanation
def pass_test_in_context(question):
    # Read the prompt from the file
    with open(in_context_prompt_file, "r") as f:
        prompt = f.read()

    regex = e_a_regex

    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    output = get_completion_text(prompt, max_tokens=100, regex=regex, temperature=0.1, min_p=0.1)

    # Extract the explanation and answer from the output, the first line is the explanation prefixed with "Explanation:" and the second line is the answer prefixed with "Evaluation:"
    explanation, answer = output.split("\nEvaluation:")

    explanation = explanation.replace("Explanation:", "").strip()
    answer = answer.strip()

    # If the answer is "Yes" return True, else return False
    return answer == "Yes", explanation


# Function to check if a question is situationally appropriate, returns True if the question is situationally appropriate, False otherwise and the explanation
def pass_test_situ_aprop(question):
    # Read the prompt from the file
    with open(situ_aprop_prompt_file, "r") as f:
        prompt = f.read()

    regex = e_a_regex

    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    output = get_completion_text(prompt, max_tokens=100, regex=regex, temperature=0.1, min_p=0.1)

    # Extract the explanation and answer from the output, the first line is the explanation prefixed with "Explanation:" and the second line is the answer prefixed with "Evaluation:"
    explanation, answer = output.split("\nEvaluation:")

    explanation = explanation.replace("Explanation:", "").strip()
    answer = answer.strip()

    # If the answer is "Yes" return True, else return False
    return answer == "Yes", explanation


# Function to check the question using all the rubrics and return a bool and an array of explanations and the individual results
def pass_test_comprehensive(question):
    in_context_result, in_context_explanation = pass_test_in_context(question)
    situ_aprop_result, situ_aprop_explanation = pass_test_situ_aprop(question)

    # Create an array of explanations and results
    output = []

    # Append the individual results and explanations
    output.append({"test": "In Context", "result": in_context_result, "explanation": in_context_explanation})
    output.append({"test": "Situationally Appropriate", "result": situ_aprop_result, "explanation": situ_aprop_explanation})

    # If both the results are True, return True, else return False
    return in_context_result and situ_aprop_result, output


# Function to check if a set of notes is useful, returns the explanation and the relevance score
def pass_test_usefulness(question, notes):
    # Read the prompt from the file
    with open(usefulness_prompt_file, "r") as f:
        prompt = f.read()

    regex = e_r_regex

    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    # Replace {{NOTES}} with the notes
    prompt = re.sub(r"{{NOTES}}", notes, prompt)

    output = get_completion_text(prompt, max_tokens=100, regex=regex, temperature=0.1, min_p=0.1)

    # Extract the explanation and answer from the output, the first line is the explanation prefixed with "Explanation:" and the second line is the answer prefixed with "Relevance score:"
    explanation, score = output.split("\nRelevance score:")

    explanation = explanation.strip()
    score = int(score.strip())

    return score, explanation