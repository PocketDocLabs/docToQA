import re
from backend.vllm import get_completion_text
from backend.vllm import token_count
import anyascii
from rich.progress import Progress
from segmentation import naive_split

summary_prompt_file = "generation_prompts/summary-llama3.txt"

update_summary_prompt_file = "generation_prompts/update-summary-llama3.txt"

combine_summaries_prompt_file = "generation_prompts/combine-summary-llama3.txt"


summary_regex = "[A-Z][a-z]([A-Za-z0-9,'\" ]|([A-Z][.])|([0-9][.]))+[.?!]"

def check_summary(text, max_tokens=512):
    # Check if there are any tokens repeated more than 3 times
    num_tokens, ids = token_count(text, send_ids=True)

    # If there are any ids repeated more than 3 times in a row in the token list, return False
    if any([ids[i] == ids[i + 1] == ids[i + 2] == ids[i + 3] for i in range(len(ids) - 3)]):
        return False
    
    if num_tokens >= max_tokens * 0.95:
        return False
    
    # If there are no tokens repeated more than 3 times, return True
    return True

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
    completion = get_completion_text(prompt, max_tokens=max_tokens, temperature=2.0, min_p=0.08, repetition_penalty=1.0, regex=regex)

    if not check_summary(completion, max_tokens=max_tokens):
        while not check_summary(completion, max_tokens=max_tokens):
            completion = get_completion_text(prompt, max_tokens=max_tokens, temperature=2.0, min_p=0.08, repetition_penalty=1.0, regex=regex)

    # Convert the completion to ASCII
    completion = anyascii.anyascii(completion)

    # Strip the completion
    completion = completion.strip()

    # Check for </summary> tag and remove it
    if completion.endswith("</summary>"):
        completion = completion[:-10]

    # Strip the completion
    completion = completion.strip()

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
    completion = get_completion_text(prompt, max_tokens=512, temperature=1.5, min_p=0.1, repetition_penalty=1.0, regex=regex)

    if not check_summary(completion, max_tokens=512):
        while not check_summary(completion, max_tokens=512):
            completion = get_completion_text(prompt, max_tokens=512, temperature=1.5, min_p=0.1, repetition_penalty=1.0, regex=regex)

    # Convert the completion to ASCII
    completion = anyascii.anyascii(completion)

    # Strip the completion
    completion = completion.strip()

    # Check for </summary> tag and remove it

    if completion.endswith("</summary>"):
        completion = completion[:-10]

    # Strip the completion
    completion = completion.strip()

    # Return the completion
    return completion
    

# Function to summarize a very long text
def long(text, verbose=False):

    segments = []
    
    # Split the text into segments using naive_split
    segments = naive_split(text, max_tokens=3000)

    print(f"Number of segments: {len(segments)}")

    # Convert the segments into an array with 2 keys, text and id
    segments = [{"text": segment, "id": i} for i, segment in enumerate(segments)]

    print("Segmentation complete")

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

    print("Summarizing...")

    # If verbose is True, use rich to display a progress bar
    # if verbose:
    #     with Progress() as progress:
    #         # Create a progress bar
    #         task = progress.add_task("[green]Summarizing...", total=len(segments))
            
    #         with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #             # Map summarize_segment to each segment
    #             results = executor.map(summarize_segment, segments)
    #             for result in results:
    #                 summaries.append(result)
    #                 # Update the progress bar each time a segment is processed
    #                 progress.update(task, advance=1)
    # else:
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #         # Map summarize_segment to each segment
    #         results = executor.map(summarize_segment, segments)
    #         for result in results:
    #             summaries.append(result)

    # Without using parallel processing
    if verbose:
        with Progress() as progress:
            # Create a progress bar
            task = progress.add_task("[green]Summarizing...", total=len(segments))
            for segment in segments:
                summary = summarize_segment(segment)
                summaries.append(summary)
                progress.update(task, advance=1)
    else:
        for segment in segments:
            summary = summarize_segment(segment)
            summaries.append(summary)

    # Sort the summaries by id
    summaries = sorted(summaries, key=lambda x: x['id'])

    # Extract the summaries from the results
    summaries = [result['summary'] for result in summaries]
        
    # use summarize.combine_summaries to combine the summaries
    combined_summary = combine(summaries)

    # Strip the combined summary
    combined_summary = combined_summary.strip()

    # Return the combined summary
    return combined_summary

