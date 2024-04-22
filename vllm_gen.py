import requests

# Generate API URL
# gen_url = "http://192.168.13.53:8080/v1/completions"
gen_url = "https://grey-cleaning-electronics-presence.trycloudflare.com/v1/completions"

# Token count API URL
# token_count_url = "http://192.168.13.53:8080/tokenize"
token_count_url = "https://grey-cleaning-electronics-presence.trycloudflare.com/v1/token/encode"

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


def get_completion(prompt, max_tokens=200, temperature=1.5, min_p=0.1, stop_sequence=[], regex=""):

    # Set the headers
    headers = {
        "X-API-KEY": "EMPTY",
    }

    # Set the JSON
    json = {
                "model": "meta-llama/Meta-Llama-3-70B-Instruct",
                "prompt": prompt,
                "max_context_length": 16000,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "min_p": min_p,
                "guided_regex": regex,
                "stop": stop_sequence
            }
    
    # Expected response
    """
    {'id': 'cmpl-3263575ebf6a4efb82d7bb8cba177a7b', 'object': 'text_completion', 'created': 844573, 'model': 'llama-3-70b-instruct', 'choices': [{'index': 0, 'text': " What is the general overview of the ignition system in a vehicle?\nClosed-ended question: Is the ignition system responsible for generating high voltage?\nSemi-Structured question: Can you explain the difference between the Transistorised Coil Ignition (TCI) system and the Motronic ignition system?\nLeading question: Don't you think that the ignition system is a critical component of a vehicle's engine?\n\nInstructions (Imperatives)\n\nShort instruction: Check the ignition system regularly to ensure proper engine performance.\nScenario", 'logprobs': None, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 9396, 'total_tokens': 9496, 'completion_tokens': 100}}
    """
    # Send the request
    response = requests.post(gen_url, headers=headers, json=json, timeout=600)

    # Return the response
    return response.json()

def get_completion_text(prompt, max_tokens=200, temperature=1.0, min_p=0.2, stop_sequence=["<|eot_id|>"], regex=""):
    return get_completion(prompt, max_tokens, temperature, min_p, stop_sequence, regex)["choices"][0]["text"]