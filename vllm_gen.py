import requests


# base_url = "http://127.0.0.1:2242"

base_url = "http://192.168.13.53:2242"

# Generate API URL

# Generate API URL from the base URL
gen_url = base_url + "/v1/completions"

# Token count API URL from the base URL
token_count_url = base_url + "/v1/token/encode"

# Model list API URL from the base URL
model_list_url = base_url + "/v1/models"


# Function to get token count
def token_count(text, send_ids=False):
    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Set the JSON
    json = {
                "prompt": text
            }

    # Send the request
    response = requests.post(token_count_url, headers=headers, json=json)

    # Expected response
    # {'value': 28, 'ids': [3957, 13465, 3958, 369, 499, 5380, 70869, 25, 8489, 433, 14117, 627, 697, 33194, 18607, 25, 220, 17, 271, 3923, 374, 279, 6864, 315, 9822, 5380, 70869, 25]}

    # Count the number of tokens
    num_tokens = response.json()["value"]

    if send_ids:
        return num_tokens, response.json()["ids"]
    else:
        return num_tokens

# Function to get the list of models
def get_model_list():
    # Send the request
    response = requests.get(model_list_url)

    # Return the response
    return response.json()

# Function to return the name of the first model
def get_first_model_name():
    # Get the list of models
    model_list = get_model_list()

    # Expected response
    # {'object': 'list', 'data': [{'id': 'Meta-Llama-3-8B-Instruct-AWQ', 'object': 'model', 'created': 1713796466, 'owned_by': 'pygmalionai', 'root': 'Meta-Llama-3-8B-Instruct-AWQ', 'parent': None, 'permission': [{'id': 'modelperm-08e05603f36a4b689afaaa806ce7bfcb', 'object': 'model_permission', 'created': 1713796466, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]}

    # Get the name of the first model
    first_model_name = model_list["data"][0]["id"]

    # Return the name of the first model
    return first_model_name


def get_completion(prompt, max_tokens=200, temperature=1.0, min_p=0.0, top_k=0, repetition_penalty=1.0, stop_sequence=[], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False):

    model = get_first_model_name()

    # Set the headers
    headers = {
        "X-API-KEY": "EMPTY",
    }

    # Set the JSON
    json = {
                "model": model,
                "prompt": prompt,
                "max_context_length": 16000,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p,
                "top_k": top_k,
                "use_beam_search": beam_search,
                "guided_regex": regex,
                "guided_grammar": grammar,
                "guided_decoding_backend": "lm-format-enforcer",
                "skip_special_tokens": skip_special_tokens,
                "stop": stop_sequence,
                "ignore_eos": ignore_eos
            }
    
    # Expected response
    """
    {'id': 'cmpl-3263575ebf6a4efb82d7bb8cba177a7b', 'object': 'text_completion', 'created': 844573, 'model': 'llama-3-70b-instruct', 'choices': [{'index': 0, 'text': " What is the general overview of the ignition system in a vehicle?\nClosed-ended question: Is the ignition system responsible for generating high voltage?\nSemi-Structured question: Can you explain the difference between the Transistorised Coil Ignition (TCI) system and the Motronic ignition system?\nLeading question: Don't you think that the ignition system is a critical component of a vehicle's engine?\n\nInstructions (Imperatives)\n\nShort instruction: Check the ignition system regularly to ensure proper engine performance.\nScenario", 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9396, 'total_tokens': 9496, 'completion_tokens': 100}}
    """
    # Send the request
    response = requests.post(gen_url, headers=headers, json=json, timeout=1800)

    # Return the response
    return response.json()

def get_completion_text(prompt, max_tokens=200, temperature=1.0, min_p=0.0, top_k=0, repetition_penalty=1.0, stop_sequence=[], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False):
    output = ""

    while output == "":
        output = get_completion(prompt, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens)["choices"][0]["text"]
        output = output.strip()
    return output