from backend.vllm import get_completion_text
import re
import concurrent.futures

from rich.progress import Progress


context_split_prompt_file = "./prep_prompts/context-split-llama3.md"


def prepare_regex(text):
    # Read the input text and remove leading/trailing whitespace
    output_regex = text.strip()

    # Replace groups of 2 or more newlines with a double newline
    output_regex = re.sub(r"(\n){2,}", r"\n\n", output_regex)

    # Replace ". " with "<DOT_SPACE>"
    output_regex = re.sub("\. ", "<DOT_SPACE>", output_regex)

    # Replace "? " with "<QUESTION_SPACE>"
    output_regex = re.sub("\? ", "<QUESTION_SPACE>", output_regex)

    # Replace "! " with "<EXCLAMATION_SPACE>"
    output_regex = re.sub("\! ", "<EXCLAMATION_SPACE>", output_regex)

    # Replace "\n\n" with "<DOUBLE_NEWLINE>"
    output_regex = re.sub(r"\n\n", "<DOUBLE_NEWLINE>", output_regex)

    # Replace "\n" with "<NEWLINE>"
    output_regex = re.sub(r"\n", "<NEWLINE>", output_regex)

    # Escape special characters
    output_regex = re.escape(output_regex)

    # Replace "<DOT_SPACE>" with ".(\n|\n\n| |\n</split>\n\n<split>\n)"
    output_regex = re.sub(r"<DOT_SPACE>", r"\.(\\n|\\n\\n| |\\n</split>\\n\\n<split>\\n)", output_regex)

    # Replace "<QUESTION_SPACE>" with "?(\n|\n\n| |\n</split>\n\n<split>\n)"
    output_regex = re.sub(r"<QUESTION_SPACE>", r"\?(\\n|\\n\\n| |\\n</split>\\n\\n<split>\\n)", output_regex)

    # Replace "<EXCLAMATION_SPACE>" with "!(\n|\n\n| |\n</split>\n\n<split>\n)"
    output_regex = re.sub(r"<EXCLAMATION_SPACE>", r"\!(\\n|\\n\\n| |\\n</split>\\n\\n<split>\\n)", output_regex)

    # Replace "<DOUBLE_NEWLINE>" with "(\n|\n\n| |\n</split>\n\n<split>\n)"
    output_regex = re.sub(r"<DOUBLE_NEWLINE>", r"(\\n|\\n\\n| |\\n</split>\\n\\n<split>\\n)", output_regex)

    # Replace "<NEWLINE>" with "(\n|\n\n| |\n</split>\n\n<split>\n)"
    output_regex = re.sub(r"<NEWLINE>", r"(\\n|\\n\\n| |\\n</split>\\n\\n<split>\\n)", output_regex)

    # Add the split tags to the beginning and end
    output_regex = "<split>\\n" + output_regex + "\\n</split>"

    # old and busted
    # Regex to match paragraphs between quadruple newlines
    output_regex = """(<split>\n(.|\s)*?\n</split>\n\n)+"""

    return output_regex


def smart_split(text):
    # Function to split text into multiple segments based on the context using a language model
    # Prepare the regex
    output_regex = prepare_regex(text)

    # Read the context split prompt from the file
    with open(context_split_prompt_file, "r") as f:
        context_split_prompt = f.read()

    # Insert the text into the context split prompt
    context_split_prompt = context_split_prompt.replace("{{text}}", text)

    # Calculate the token limit by dividing the character count by 4 and multiplying by 1.1
    token_limit = int(len(text) / 4 * 1.1)

    output = get_completion_text(context_split_prompt, regex=output_regex, max_tokens=token_limit, temperature=0.5, min_p=0.08, ignore_eos=True, skip_special_tokens=True)

    # The output will have text enclosed in <split></split> tags, so we need to read the text between the tags to a list using regex
    output_list = re.findall(r"<split>(.*?)</split>", output, re.DOTALL)

    # Strip leading/trailing whitespace from each element in the list
    output_list = [x.strip() for x in output_list]

    return output_list

def naive_split(text, max_tokens=4000, overlap=0.2):
    # Split the text into multiple segments based on the max tokens limit and overlap. For performance reasons we will assume 1 token is 4 characters
    # Calculate the number of tokens to overlap
    overlap_tokens = int(max_tokens * overlap)

    overlap_tokens = overlap_tokens * 4

    max_tokens = max_tokens * 4

    # Every max_tokens - overlap_tokens characters, we will take the next max_tokens characters rounded to the previous punctuation mark
    # Initialize the start and end indices
    start_index = 0
    end_index = 0

    # Initialize the output list
    output_list = []

    # Loop until the end index is equal to the length of the text
    while end_index < len(text):
        # Calculate the end index
        end_index = start_index + max_tokens

        # If the end index is greater than the length of the text, then set it to the length of the text
        if end_index > len(text):
            end_index = len(text)

        # If the end index is equal to the length of the text, then break
        if end_index == len(text):
            break

        # If the end index is less than the length of the text, then find the last ". ", "? ", or "! " before the end index and set the end index to that index
        end_index = text.rfind(" ", start_index, end_index)

        # If the end index is still -1, then set it to the start index + max tokens
        if end_index == -1:
            end_index = start_index + max_tokens

        # Append the text from start index to end index to the output list
        output_list.append(text[start_index:end_index])

        # Update the start index to the end index - overlap tokens
        start_index = end_index - overlap_tokens

        # Find the last newline or whitespace character before the start index
        start_index = text.rfind(" ", 0, start_index)

        if start_index == -1:
            start_index = text.rfind("\n", 0, start_index)

        # If the start index is still -1, then set it to end index - overlap tokens
        if start_index == -1:
            start_index = end_index - overlap_tokens

    # Append the remaining text to the output list
    output_list.append(text[start_index:])

    # Strip leading/trailing whitespace from each element in the list
    output_list = [x.strip() for x in output_list]
    
    return output_list

def smart_split_long(text, verbose=False):
    # Use naive_split to split the text into 4000 token segments
    segments = naive_split(text, max_tokens=4000)

    # Initialize the output list
    output_list = []
    if verbose:
        with Progress() as progress:
            # Create a progress bar
            task = progress.add_task("[green]Segmenting...", total=len(segments))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Map and process segments
                for output in executor.map(smart_split, segments):
                    output_list.extend(output)
                    progress.update(task, advance=1)
        
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Map and process segments
            for output in executor.map(smart_split, segments):
                output_list.extend(output)

    # Return the output list
    return output_list