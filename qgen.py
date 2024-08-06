import json
import random
import re

import anyascii

import concurrent.futures

from rich.progress import Progress

import evaluations
import summary

from segmentation import naive_split


from backend.vllm import get_completion_text



# Prompt file
prompt_file = "./generation_prompts/q-gen-llama3.txt"



def construct_regex():
    # Construct the regex
    
    # line
    line = "[^\r\n\x0b\x0c\x85\u2028\u2029|\"0-9][^\r\n\x0b\x0c\x85\u2028\u2029|]+"

    # question
    question = " " + line + "\n"

    # questionsubgroup
    questionsubgroup = "Questions (Interrogatives)\n\nOpen-ended question:" + question + "Closed-ended question:" + question + "Semi-Structured question:" + question + "Leading question:" + question + "\n\nInstructions (Imperatives)\n\nShort instruction:" + question + "Scenario-based instruction:" + question + "Problem-based instruction:" + question + "\n\nPrompts\n\nShort prompt:" + question + "Scenario-based prompt:" + question + "Problem-based prompt:" + question + "\n\nRequests (Modal Constructions)\n\nFormal request:" + question + "Informal request:" + question + "Polite request:" + question + "Direct request:" + question

    # initial group
    initial_group = question + "Closed-ended question:" + question + "Semi-Structured question:" + question + "Leading question:" + question + "\n\nInstructions (Imperatives)\n\nShort instruction:" + question + "Scenario-based instruction:" + question + "Problem-based instruction:" + question + "\n\nPrompts\n\nShort prompt:" + question + "Scenario-based prompt:" + question + "Problem-based prompt:" + question + "\n\nRequests (Modal Constructions)\n\nFormal request:" + question + "Informal request:" + question + "Polite request:" + question + "Direct request:" + question + "\n\nCategory: Detailed\n\n\n" + questionsubgroup + "\n\nCategory: Difficult or uncomfortable\n\n\n" + questionsubgroup + "\n\nFinal comments:"

    # return the regex
    return initial_group


def construct_regex_alt():
    # Construct the regex
    
    # line
    line = "[^\r\n\x0b\x0c\x85\u2028\u2029|\"0-9][^\r\n\x0b\x0c\x85\u2028\u2029|]+"

    # question
    question = " " + line + "\n"

    # questionsubgroup
    questionsubgroup = "Questions (Interrogatives)\n\nOpen-ended question:" + question + "Closed-ended question:" + question + "Semi-Structured question:" + question + "Leading question:" + question + "\n\nInstructions (Imperatives)\n\nShort instruction:" + question + "Scenario-based instruction:" + question + "Problem-based instruction:" + question + "\n\nPrompts\n\nShort prompt:" + question + "Scenario-based prompt:" + question + "Problem-based prompt:" + question + "\n\nRequests (Modal Constructions)\n\nFormal request:" + question + "Informal request:" + question + "Polite request:" + question + "Direct request:" + question

    # initial group
    initial_group = question + "Closed-ended question:" + question + "Semi-Structured question:" + question + "Leading question:" + question + "\n\nInstructions (Imperatives)\n\nShort instruction:" + question + "Scenario-based instruction:" + question + "Problem-based instruction:" + question + "\n\nPrompts\n\nShort prompt:" + question + "Scenario-based prompt:" + question + "Problem-based prompt:" + question + "\n\nRequests (Modal Constructions)\n\nFormal request:" + question + "Informal request:" + question + "Polite request:" + question + "Direct request:" + question + "\n\nCategory: Analytical\n\n\n" + questionsubgroup + "\n\nCategory: Practical\n\n\n" + questionsubgroup + "\n\nFinal comments:"

    # return the regex
    return initial_group


# Construct the regex
regex_alt = construct_regex()
regex = construct_regex_alt()


# Function to clean up text before generating questions or summaries
def clean_text(text):
    # Convert the text to ASCII
    text = anyascii.anyascii(text)

    # Split the text into lines
    lines = text.split("\n")

    # Replace whitespace with a single space
    for line in lines:
        line = re.sub(r"\s", " ", line)

    # Join the lines back together
    text = "\n".join(lines)

    # Strip the text
    text = text.strip()

    # Merge groups of newlines greater than 2 into 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove any space before a newline
    text = re.sub(r" +\n", "\n", text)

    # Merge groups of spaces into a single space
    text = re.sub(r" +", " ", text)

    # Merge groups of repeated characters 5 or longer into 4 characters
    text = re.sub(r"(.{4})\1{2,}", r"\1\1\1", text)

    # Strip the text again
    text = text.strip()

    # Return the cleaned text
    return text


# evaluation.pass_test_comprehensive(question) returns a list of evaluations for the question and an array of the evaluations performed
# For each question check it with evaluations.pass_test_comprehensive(question) and create a json object with the questions and the returned array of evaluations and print it
def process_question(question):
    passed, evaluations_performed = evaluations.pass_test_comprehensive(question)
    return {
        "question": question,
        "passed": passed,
        "evaluations_performed": evaluations_performed
    }


# Function to generate a list of questions given a text and a summary
def generate_questions_rough(text, summary):
    # Read the prompt from the file
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Replace {{DOCUMENT}} with the text
    prompt = re.sub(r"{{DOCUMENT}}", text, prompt)

    # Replace {{DOCUMENT_SUMMARY}} with the summary
    prompt = re.sub(r"{{DOCUMENT_SUMMARY}}", summary, prompt)

    # Use get_completion_text to generate questions
    outputs = get_completion_text(prompt, max_tokens=2500, regex=regex, temperature=2.5, min_p=0.3, repetition_penalty=1.02, num_results=1)

    # If outputs is a string, convert it to a list
    if isinstance(outputs, str):
        outputs = [outputs]

    # List of questions
    questions = []

    for output in outputs:
        # Split the output into lines
        lines = output.split("\n")

        # Remove empty lines
        lines = [line for line in lines if line]

        # List of prefixes to look for
        prefixes = ["Open-ended question:", "Closed-ended question:", "Semi-Structured question:", "Leading question:", "Short instruction:", "Scenario-based instruction:", "Problem-based instruction:", "Short prompt:", "Scenario-based prompt:", "Problem-based prompt:", "Formal request:", "Informal request:", "Polite request:", "Direct request:"]

        for line in lines:
            # Check if the line starts with a prefix
            if any([line.startswith(prefix) for prefix in prefixes]):
                # Strip the prefix
                line = line.split(":")[1]

                # Strip the line
                line = line.strip()

                questions.append(line)

    # Clean the questions
    questions = [clean_text(question) for question in questions]

    return questions


def generate_questions_rough_alt(text, summary):
    # Read the prompt from the file
    with open(prompt_file, "r") as f:
        prompt = f.read()

    # Replace {{DOCUMENT}} with the text
    prompt = re.sub(r"{{DOCUMENT}}", text, prompt)

    # Replace {{DOCUMENT_SUMMARY}} with the summary
    prompt = re.sub(r"{{DOCUMENT_SUMMARY}}", summary, prompt)

    # Use get_completion_text to generate questions
    outputs = get_completion_text(prompt, max_tokens=2500, regex=regex_alt, temperature=1.5, min_p=0.2, num_results=1)

    # If outputs is a string, convert it to a list
    if isinstance(outputs, str):
        outputs = [outputs]

    # List of questions
    questions = []

    for output in outputs:
        # Split the output into lines
        lines = output.split("\n")

        # Remove empty lines
        lines = [line for line in lines if line]

        # List of prefixes to look for
        prefixes = ["Open-ended question:", "Closed-ended question:", "Semi-Structured question:", "Leading question:", "Short instruction:", "Scenario-based instruction:", "Problem-based instruction:", "Short prompt:", "Scenario-based prompt:", "Problem-based prompt:", "Formal request:", "Informal request:", "Polite request:", "Direct request:"]

        for line in lines:
            # Check if the line starts with a prefix
            if any([line.startswith(prefix) for prefix in prefixes]):
                # Strip the prefix
                line = line.split(":")[1]

                # Strip the line
                line = line.strip()

                questions.append(line)

    # Clean the questions
    questions = [clean_text(question) for question in questions]

    return questions


# Function to generate a list of questions given a text
def generate_questions(text, summary, verbose=False):
    # Generate the questions
    questions = generate_questions_rough(text, summary)

    question_evaluation_results = []

    # Evaluate the questions
    if verbose:
        # use rich progress bar and concurrent futures to process questions
        with Progress() as progress:
            # Create a task for the progress bar
            task = progress.add_task("[green]Processing questions...", total=len(questions))
            
            # Use ThreadPoolExecutor to process 1 question at a time
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                # Map and process questions, update the progress bar upon each completion
                for question_evaluation in executor.map(process_question, questions):
                    question_evaluation_results.append(question_evaluation)
                    progress.update(task, advance=1)  # Advance the progress bar

    else:
        # Process questions without a progress bar using concurrent futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            question_evaluation_results = list(executor.map(process_question, questions))

    output_list = []

    # Append the questions that passed all the evaluations to the output list
    for question_evaluation in question_evaluation_results:
        if question_evaluation["passed"]:
            output_list.append(question_evaluation["question"])

    return output_list


# function to generate questions from a long segment of text that pairs the questions with the text in an object and returns a list of these objects
# Format: {"text": segment, "questions": [questions]}
def generate_questions_long(text, summary, verbose=False):
    # Clean the text
    text = clean_text(text)

    # Split the text into segments
    segments = naive_split(text, max_tokens=1000)

    # List of questions
    output_objects = []

    # Generate questions for each segment
    if verbose:
        with Progress() as progress:
            # Create a progress bar
            task = progress.add_task("[green]Generating questions...", total=len(segments))
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Map and process segments, update the progress bar upon each completion
                for output_object in executor.map(lambda segment: {"text": segment, "questions": generate_questions(segment, summary, verbose=False)}, segments):
                    output_objects.append(output_object)
                    progress.update(task, advance=1)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            output_objects = list(executor.map(lambda segment: {"text": segment, "questions": generate_questions(segment, summary, verbose=False)}, segments))

    return output_objects