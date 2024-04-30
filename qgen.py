import json
import random
import re

import anyascii

import concurrent.futures

from rich.progress import Progress

import evaluations
import summary

from vllm_gen import get_completion
from vllm_gen import get_completion_text
from vllm_gen import token_count


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
    initial_group = question + "Closed-ended question:" + question + "Semi-Structured question:" + question + "Leading question:" + question + "\n\nInstructions (Imperatives)\n\nShort instruction:" + question + "Scenario-based instruction:" + question + "Problem-based instruction:" + question + "\n\nPrompts\n\nShort prompt:" + question + "Scenario-based prompt:" + question + "Problem-based prompt:" + question + "\n\nRequests (Modal Constructions)\n\nFormal request:" + question + "Informal request:" + question + "Polite request:" + question + "Direct request:" + question + "\n\nCategory: Detailed\n\n\n" + questionsubgroup + "\n\nCategory: Not directly related\n\n\n" + questionsubgroup + "\n\nFinal comments:"

    # return the regex
    return initial_group


# Construct the regex
regex = construct_regex()


# Function to clean up text before generating questions or summaries
def clean_text(text):
    # Convert the text to ASCII
    text = anyascii.anyascii(text)

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
    output = get_completion_text(prompt, max_tokens=2500, regex=regex, temperature=1.5, min_p=0.1, repetition_penalty=1.05)

    # Split the output into lines
    lines = output.split("\n")

    # Remove empty lines
    lines = [line for line in lines if line]

    # List of questions
    questions = []

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
            
            # Use ThreadPoolExecutor to process 16 questions at a time
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map and process questions, update the progress bar upon each completion
                for question_evaluation in executor.map(process_question, questions):
                    question_evaluation_results.append(question_evaluation)
                    progress.update(task, advance=1)  # Advance the progress bar

    else:
        # Process questions without a progress bar using concurrent futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            question_evaluation_results = list(executor.map(process_question, questions))

    output_list = []

    # Append the questions that passed all the evaluations to the output list
    for question_evaluation in question_evaluation_results:
        if question_evaluation["passed"]:
            output_list.append(question_evaluation["question"])

    return output_list


# Function to split a string into segments of equal length 
def split_text(text, max_length):
    # List of segments
    segments = []

    # Find the length each segment should be by dividing the length of the text by the maximum segment length and rounding up then dividing the length of the text by the number of segments
    segment_length = -(-len(text) // max_length)

    # Split the text into segments
    for i in range(segment_length):
        segments.append(text[i * max_length:(i + 1) * max_length])

    return segments


# Function to generate questions from a long segment of text
def generate_questions_long(text, verbose=False):
    # Clean the text
    text = clean_text(text)

    # Get the summary of the text
    doc_summary = summary.long(text, verbose=verbose)

    # Split the text into segments
    segments = split_text(text, 20000)

    # List of questions
    questions = []

    # Generate questions for each segment
    for segment in segments:
        questions += generate_questions(segment, doc_summary, verbose=verbose)

    return questions