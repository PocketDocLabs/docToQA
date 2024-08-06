import requests


# base_url = "http://127.0.0.1:2242"

# base_url = "http://192.168.13.53:2242"

# Url list
url_list = [
    # "http://0.0.0.0:7860"
    "https://jsm67wjxxu0o1k-8000.proxy.runpod.net/"
]


# Generate API URL
gen_url = "/v1/completions"

# Token count API URL
token_count_url = "/v1/token/encode"

# Model list API URL
model_list_url = "/v1/models"

# Metrics API URL
metrics_url = "/metrics"


# Function to get server metrics
def get_metrics(url=None):
    # Set the URL
    url = url + metrics_url


    # Send the request
    response = requests.get(url)

    # Should be in prometheus format
    # Example:
    # HELP aphrodite:num_requests_running Number of requests currently running on GPU.
    # TYPE aphrodite:num_requests_running gauge
    # aphrodite:num_requests_running{model_name="./Models/llama3-8b-instruct-exl2-6.0bpw"} 1.0
    # HELP aphrodite:num_requests_swapped Number of requests swapped to CPU.
    # TYPE aphrodite:num_requests_swapped gauge
    # aphrodite:num_requests_swapped{model_name="./Models/llama3-8b-instruct-exl2-6.0bpw"} 0.0
    # HELP aphrodite:num_requests_waiting Number of requests waiting to be processed.
    # TYPE aphrodite:num_requests_waiting gauge
    # aphrodite:num_requests_waiting{model_name="./Models/llama3-8b-instruct-exl2-6.0bpw"} 0.0

    return response

# Function to determin which url has the fewest requests running or waiting
def get_best_url(url_list):
    # Get the metrics for each URL
    url_metrics = {}
    for url in url_list:
        try:
            response = get_metrics(url)
            url_metrics[url] = response.text
        except:
            # Remove the URL from the list if there is an error
            url_list.remove(url)

            

    # Parse the metrics to get the number of requests running and waiting
    num_requests_running = {}
    num_requests_waiting = {}
    for url, metrics in url_metrics.items():
        for line in metrics.split("\n"):
            if "aphrodite:num_requests_running{" in line:
                num_requests_running[url] = int(line.split(" ")[-1].split(".")[0])
            if "aphrodite:num_requests_waiting{" in line:
                num_requests_waiting[url] = int(line.split(" ")[-1].split(".")[0])

    # Find the URL with the fewest requests running or waiting, if there is a tie, use the first one
    best_url = None
    best_num_requests = None
    for url in url_list:
        num_requests = num_requests_running.get(url, 0) + num_requests_waiting.get(url, 0)
        if best_url is None or num_requests < best_num_requests:
            best_url = url
            best_num_requests = num_requests

    return best_url



# Function to get token count
def token_count(text, send_ids=False, url=None):
    if url is None:
        url = get_best_url(url_list)

    # Set the URL
    req_url = url + token_count_url

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Set the JSON
    json = {
                "prompt": text
            }

    # Send the request
    response = requests.post(req_url, headers=headers, json=json)

    # Expected response
    # {'value': 28, 'ids': [3957, 13465, 3958, 369, 499, 5380, 70869, 25, 8489, 433, 14117, 627, 697, 33194, 18607, 25, 220, 17, 271, 3923, 374, 279, 6864, 315, 9822, 5380, 70869, 25]}

    # Count the number of tokens
    num_tokens = response.json()["value"]

    if send_ids:
        return num_tokens, response.json()["ids"]
    else:
        return num_tokens

# Function to get the list of models
def get_model_list(url=None):
    # Set the URL
    req_url = url + model_list_url

    # Send the request
    response = requests.get(req_url)

    # Return the response
    return response.json()

# Function to return the name of the first model
def get_first_model_name(url=None):

    # Get the list of models
    model_list = get_model_list(url)

    # Expected response
    # {'object': 'list', 'data': [{'id': 'Meta-Llama-3-8B-Instruct-AWQ', 'object': 'model', 'created': 1713796466, 'owned_by': 'pygmalionai', 'root': 'Meta-Llama-3-8B-Instruct-AWQ', 'parent': None, 'permission': [{'id': 'modelperm-08e05603f36a4b689afaaa806ce7bfcb', 'object': 'model_permission', 'created': 1713796466, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}]}]}

    # Get the name of the first model
    first_model_name = model_list["data"][0]["id"]

    # Return the name of the first model
    return first_model_name


def get_completion(prompt, max_tokens=200, temperature=1.0, min_p=0.0, top_k=-1, repetition_penalty=1.0, stop_sequence=["<|end_of_text|>", "<|eot_id|>"], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False, num_results=1, best_of=1, url=None):

    model = get_first_model_name(url)

    # Set the URL
    req_url = url + gen_url

    # if n is greater than best_of, then set best_of to n
    if num_results > best_of:
        best_of = num_results

    # Set the headers
    headers = {
        "X-API-KEY": "EMPTY",
    }

    # Set the JSON
    json = {
                "model": model,
                "prompt": prompt,
                "n": num_results,
                "best_of": best_of,
                "max_context_length": 11000,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
                "min_p": min_p,
                "top_k": top_k,
                "use_beam_search": beam_search,
                "guided_regex": regex,
                "guided_grammar": grammar,
                "guided_decoding_backend": "outlines",
                "skip_special_tokens": skip_special_tokens,
                "stop": stop_sequence,
                "ignore_eos": ignore_eos
            }
    
    # Expected response
    """
    {'id': 'cmpl-3263575ebf6a4efb82d7bb8cba177a7b', 'object': 'text_completion', 'created': 844573, 'model': 'llama-3-70b-instruct', 'choices': [{'index': 0, 'text': " What is the general overview of the ignition system in a vehicle?\nClosed-ended question: Is the ignition system responsible for generating high voltage?\nSemi-Structured question: Can you explain the difference between the Transistorised Coil Ignition (TCI) system and the Motronic ignition system?\nLeading question: Don't you think that the ignition system is a critical component of a vehicle's engine?\n\nInstructions (Imperatives)\n\nShort instruction: Check the ignition system regularly to ensure proper engine performance.\nScenario", 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9396, 'total_tokens': 9496, 'completion_tokens': 100}}
    """
    # Send the request
    response = requests.post(req_url, headers=headers, json=json, timeout=1800)

    # Return the response
    return response.json()

def get_completion_text(prompt, max_tokens=200, temperature=1.0, min_p=0.0, top_k=-1, repetition_penalty=1.0, stop_sequence=["<|end_of_text|>", "<|eot_id|>"], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False, num_results=1, best_of=1):

    # Get the best URL
    url = get_best_url(url_list)

    if num_results == 1:
        output = ""

        while output == "":
            response = get_completion(prompt, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results, best_of, url)

            try:
                output = response["choices"][0]["text"]
            except KeyError:
                # Raise an error with the response
                raise ValueError(response)

        return output.strip()
    
    else:
        outputs = []

        response = get_completion(prompt, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results, best_of, url)

        try:
            for choice in response["choices"]:
                outputs.append(choice["text"].strip())
        except KeyError:
            # Raise an error with the response
            raise ValueError(response)
        # Remove empty strings
        outputs = [x for x in outputs if x]

        # If the number of outputs is less than the number of results, then get more completions
        while len(outputs) < num_results:
            response = get_completion(prompt, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results - len(outputs), best_of, url)

            for choice in response["choices"]:
                outputs.append(choice["text"].strip())

            # Remove empty strings
            outputs = [x for x in outputs if x]

        return outputs
    