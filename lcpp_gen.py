import requests

# Generate API URL
# gen_url = "http://192.168.13.53:8080/v1/completions"
gen_url = "http://127.0.0.1:8080/v1/completions"

# Token count API URL
# token_count_url = "http://192.168.13.53:8080/tokenize"
token_count_url = "http://127.0.0.1:8080/tokenize"

def token_count(text, send_ids=False):
    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Set the JSON
    json = {
                "content": text
            }

    # Send the request
    response = requests.post(token_count_url, headers=headers, json=json)

    # Expected response
    # {'tokens': [415, 2936, 9060, 285, 1142, 461, 10575, 754, 272, 17898, 3914, 28723]}

    # Count the number of tokens
    num_tokens = len(response.json()["tokens"])

    if send_ids:
        return num_tokens, response.json()["tokens"]
    else:
        return num_tokens


def get_completion(prompt, max_tokens=200, temperature=1.5, min_p=0.1, stop_sequence=[], grammar=""):

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Set the JSON
    json = {
                "prompt": prompt,
                "max_context_length": 16000,
                "max_length": max_tokens,
                "rep_pen": 1.0,
                "rep_pen_range": 600,
                "rep_pen_slope": 0,
                "temperature": temperature,
                "min_p": min_p,
                "sampler_order": [6, 0, 1, 2, 3, 4, 5],
                "grammar": grammar,
                "stop_sequence": stop_sequence
            }
    
    # Expected response
    """
    {'results': [{'text': '\n1. JavaScript: This language is the most popular among developers and continues to be in high demand'}]}
    """

    # Send the request
    response = requests.post(gen_url, headers=headers, json=json)

    # Return the response
    return response.json()

def get_completion_text(prompt, max_tokens=200, temperature=1.0, min_p=0.2, stop_sequence=[], grammar=""):
    return get_completion(prompt, max_tokens, temperature, min_p, stop_sequence, grammar)["content"]