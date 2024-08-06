from backend.vllm import get_completion_text
import re
import concurrent.futures

from rich.progress import Progress

import tiktoken
import semchunk


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
    encoder = tiktoken.encoding_for_model('gpt-4')
    token_counter = lambda text: len(encoder.encode(text))

    # Split the text
    output_list = semchunk.chunk(text, chunk_size=max_tokens, token_counter=token_counter)

    # If the last line of a string in output_list starts with a # and its not the last chunk, remove it and prepend it to the next chunk
    for i in range(len(output_list) - 1):
        # Split the last line of the string
        last_line = output_list[i].split("\n")[-1]

        # If the last line starts with a # remove it and prepend it to the next chunk
        if last_line.startswith("#"):
            # Remove the last line from the string
            output_list[i] = "\n".join(output_list[i].split("\n")[:-1])

            # Prepend the last line to the next string
            output_list[i + 1] = last_line + "\n" + output_list[i + 1]

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