import re
from vllm_gen import get_completion_text
import anyascii
from rich.progress import Progress
import concurrent.futures

summary_prompt_file = "generation_prompts/summary-llama3.txt"

update_summary_prompt_file = "generation_prompts/update-summary-llama3.txt"

combine_summaries_prompt_file = "generation_prompts/combine-summary-llama3.txt"

# Explantion, answer grammar
summary_grammar = """root ::= answer

answer ::= sentence
sentence ::= start initial_content end

start ::= [A-Z]

initial_content ::= content {content}

content ::= letter | comma | apostrophe | quotation | space | initial

letter ::= [A-Za-z]
comma ::= ','
semicolon ::= ';'
apostrophe ::= '''
quotation ::= '"' | '\"'
space ::= ' '
initial ::= [A-Z] '.'

end ::= '.' | '?' | '!'
"""

summary_regex = "[A-Z][a-z]([A-Za-z0-9,'\" ]|([A-Z][.])|([0-9][.]))+[.?!]"

# Function to summarize the text
def single(text, length=2):
    # Read the prompt from the file
    with open(summary_prompt_file, "r") as f:
        prompt = f.read()

    # Replace {{text}} with the input text
    prompt = prompt.replace("{{text}}", text)

    # Replace {{sentence_count}} with the length
    prompt = prompt.replace("{{sentence_count}}", str(length))

    # If length is greater than 1 construct the regex based on the length, e.g. "([A-Z]([A-Za-z,;'\" ]|([A-Z][.]))+[.?!]) ([A-Z]([A-Za-z,;'\" ]|([A-Z][.]))+[.?!])" for length 2
    if length > 1:
        regex = " ".join([summary_regex] * length)
    else:
        regex = summary_regex

    max_tokens = length * 125

    # Get the completion text
    completion = get_completion_text(prompt, max_tokens=max_tokens, temperature=2.0, min_p=0.1, repetition_penalty=1.0, regex=regex)

    # Convert the completion to ASCII
    completion = anyascii.anyascii(completion)

    # Strip the completion
    completion = completion.strip()

    # Check for </summary> tag and remove it
    if completion.endswith("</summary>"):
        completion = completion[:-10]

    # Return the completion
    return completion

def combine(summary_list):
    # Read the prompt from the file
    with open(combine_summaries_prompt_file, "r") as f:
        prompt = f.read()

    # Wrap the first summary in <priority> tags
    summary_list[0] = f"<priority>\n{summary_list[0]}\n</priority>"

    # Join the summaries with a double newline
    summaries = "\n\n".join(summary_list)

    # Replace {{text}} with the joined summaries
    prompt = prompt.replace("{{text}}", summaries)

    # Construct the regex to have 5 sentences
    regex = " ".join([summary_regex] * 5)

    # Get the completion text
    completion = get_completion_text(prompt, max_tokens=512, temperature=0.3, min_p=0.1, repetition_penalty=1.0, regex=regex)

    # Convert the completion to ASCII
    completion = anyascii.anyascii(completion)

    # Strip the completion
    completion = completion.strip()

    # Check for </summary> tag and remove it

    if completion.endswith("</summary>"):
        completion = completion[:-10]

    # Return the completion
    return completion
    

# Function to summarize a very long text
def long(text, verbose=False):

    # create equal segments no longer than 10000 characters
    segments = []
    
    # Define the maximum segment length
    max_segment_length = 10000

    # Find the length each segment should be by dividing the length of the text by the maximum segment length and rounding up then dividing the length of the text by the number of segments
    segment_length = -(-len(text) // max_segment_length)

    # Split the text into segments
    for i in range(segment_length):
        segments.append(text[i * max_segment_length:(i + 1) * max_segment_length])

    # Convert the segments into an array with 2 keys, text and id
    segments = [{"text": segment, "id": i} for i, segment in enumerate(segments)]

    # summarize.summary(segment, length=3) will summarize the segment using a length of 3
    # Summarize the first and last 3 segments using a length of 3 and the rest using a length of 1

    # Define the summarization function with the correct length based on the segment's position
    def summarize_segment(segment):
        i = segment['id']
        text = segment['text']
        if i < 2 or i >= len(segments) - 2:
            # Summarize the first and last 3 segments with a length of 4
            summary = single(text, length=4)
        else:
            # Summarize the rest with a length of 1
            summary = single(text, length=1)
        return {"id": i, "summary": summary}

    # Use ThreadPoolExecutor to process summaries in parallel and track progress with rich
    summaries = []

    # If verbose is True, use rich to display a progress bar
    if verbose:
        with Progress() as progress:
            # Create a progress bar
            task = progress.add_task("[green]Summarizing...", total=len(segments))
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Map summarize_segment to each segment
                results = executor.map(summarize_segment, segments)
                for result in results:
                    summaries.append(result)
                    # Update the progress bar each time a segment is processed
                    progress.update(task, advance=1)
    else:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Map summarize_segment to each segment
            results = executor.map(summarize_segment, segments)
            for result in results:
                summaries.append(result)

    # Sort the summaries by id
    summaries = sorted(summaries, key=lambda x: x['id'])

    # Extract the summaries from the results
    summaries = [result['summary'] for result in summaries]
        
    # use summarize.combine_summaries to combine the summaries
    combined_summary = combine(summaries)

    # Return the combined summary
    return combined_summary

