import re
from backend.vllm import get_completion_text

import concurrent.futures

from rich.progress import Progress



in_context_prompt_file = "./rubric_prompts/out-of-context-llama3.txt"

situ_aprop_prompt_file = "./rubric_prompts/situationally-appropriate-llama3.txt"

usefulness_prompt_file = "./context_prompts/usefulness-check-llama3.txt"

enough_to_answer_prompt_file = "./agen_prompts/enough-to-answer-llama3.txt"



# Explantion, answer regex
e_a_regex = """ [^\r\n\x0b\x0c\x85\u2028\u2029]+\nEvaluation: (Yes|No)"""

# Explanation, relevance score regex with the relevance score being an integer between 0 and 10
e_r_regex = """ [^\r\n\x0b\x0c\x85\u2028\u2029]+\nRelevance score: (10|9|8|7|6|5|4|3|2|1|0)"""

# Explanation, confidence score regex with the confidence score being an integer between 0 and 10
e_c_regex = """ [^\r\n\x0b\x0c\x85\u2028\u2029]+\nConfidence score: (10|9|8|7|6|5|4|3|2|1|0)"""


# Function to check if a question is in context, returns True if the question is in context, False otherwise and the explanation
def pass_test_in_context(question):
    # Read the prompt from the file
    with open(in_context_prompt_file, "r") as f:
        prompt = f.read()

    regex = e_a_regex

    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    output = get_completion_text(prompt, max_tokens=250, regex=e_a_regex, top_k=1)

    # Extract the explanation and answer from the output, the first line is the explanation and the second line is the answer prefixed with "Evaluation:"
    explanation, answer = output.split("\nEvaluation:")

    explanation = explanation.strip()

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

    output = get_completion_text(prompt, max_tokens=250, regex=e_a_regex, temperature=0.5, min_p=0.01)

    # Extract the explanation and answer from the output, the first line is the explanation and the second line is the answer prefixed with "Evaluation:"
    explanation, answer = output.split("\nEvaluation:")

    explanation = explanation.strip()

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


# Function to check if a set of notes is useful, returns the text and the relevance score
def pass_test_usefulness(question, text):
    # Read the prompt from the file
    with open(usefulness_prompt_file, "r") as f:
        prompt = f.read()


    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    # Replace {{NOTES}} with the notes
    prompt = re.sub(r"{{NOTES}}", text, prompt)

    output = get_completion_text(prompt, max_tokens=100, regex=e_r_regex, top_k=1)

    # Extract the explanation and answer from the output, the first line is the explanation prefixed with "Explanation:" and the second line is the answer prefixed with "Relevance score:"
    explanation, score = output.split("\nRelevance score:")

    # attempt = 0
    # # if the score is not an 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10, keep asking for a score
    # while score.strip() not in ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]:
    #     output = get_completion_text(prompt, max_tokens=100, regex=e_r_regex, temperature=0.1+(attempt/20), min_p=0.01)
    #     explanation, score = output.split("\nRelevance score:")
    #     attempt += 1

    explanation = explanation.strip()

    score = int(score.strip())

    return score, text

def pass_test_usefulness_list(question, text_list, verbose=False):
    # Create an array to store the output
    output = []

    # for text in text_list:
    #     score = pass_test_usefulness(question, text)
    #     # Create an array of scores and their associated text
    #     output.append({
    #         "text": text,
    #         "score": score
    #     })

    if verbose:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a progress bar
            progress = Progress()
            task = progress.add_task("[cyan]Processing...", total=len(text_list))

            # Create a list of futures
            futures = []

            for text in text_list:
                futures.append(executor.submit(pass_test_usefulness, question, text))

            for future in concurrent.futures.as_completed(futures):
                score, text = future.result()
                # Create an array of scores and their associated text
                output.append({
                    "text": text,
                    "score": score
                })

                # Update the progress bar
                progress.update(task, advance=1)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Create a list of futures
            futures = [executor.submit(pass_test_usefulness, question, text) for text in text_list]

            # Get the results
            results = concurrent.futures.as_completed(futures)

            # Append the results to the output
            for result in results:
                score, text = result.result()
                # Create an array of scores and their associated text
                output.append({
                    "text": text,
                    "score": score
                })

    # Sort the output by the score in descending order
    output = sorted(output, key=lambda x: x["score"], reverse=True)

    return output
        
# Function to check if a set of notes is enough to answer a question, returns the text and 
def pass_test_enough_to_answer(question, text):
    # Read the prompt from the file
    with open(enough_to_answer_prompt_file, "r") as f:
        prompt = f.read()

    # Replace {{QUESTION}} with the question
    prompt = re.sub(r"{{QUESTION}}", question, prompt)

    # Replace {{NOTES}} with the notes
    prompt = re.sub(r"{{NOTES}}", text, prompt)

    output = get_completion_text(prompt, max_tokens=100, regex=e_c_regex, top_k=1)

    # Extract the explanation and answer from the output, the first line is the explanation prefixed with "Explanation:" and the second line is the answer prefixed with "Confidence score:"
    explanation, score = output.split("\nConfidence score:")

    # attempt = 0
    # # if the score is not an 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10, keep asking for a score
    # while score.strip() not in ["10", "9", "8", "7", "6", "5", "4", "3", "2", "1", "0"]:
    #     output = get_completion_text(prompt, max_tokens=100, regex=e_c_regex, temperature=0.1+(attempt/20), min_p=0.01)
    #     explanation, score = output.split("\nConfidence score:")
    #     attempt += 1

    explanation = explanation.strip()

    score = int(score.strip())

    return score, text