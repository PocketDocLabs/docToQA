import requests

# Generate API URL
# gen_url = "http://192.168.13.53:8080/v1/completions"
gen_url = "http://127.0.0.1:5000/v1/completions"

# Token count API URL
# token_count_url = "http://192.168.13.53:8080/tokenize"
token_count_url = "http://127.0.0.1:5000/v1/token/encode"

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
                "temperature": temperature,
                "min_p": min_p,
                "sampler_order": [6, 0, 1, 2, 3, 4, 5],
                "grammar_string": grammar,
                "stop": stop_sequence
            }
    
    # Expected response
    """
    {'id': 'cmpl-08ebfa49db444a4f959d6f18b2a996c7', 'choices': [{'index': 0, 'finish_reason': 'stop', 'logprobs': None, 'text': " Python has a design philosophy which emphasizes code readability, and its syntax allows programmers to express concepts in fewer lines of code than might be possible in languages such as C++ or Java.\n\nPython's simple, easy-to-understand design and syntax have made it very popular for beginners learning to code, and it has a large, supportive community. Python is also used extensively in scientific and data analysis applications.\n\nPython is an interpreted language, meaning that the code is executed directly, without the need for compilation. This makes it easier to get started with, as you don't need to install a compiler.\n\nPython has a large standard library, which includes modules for various tasks, such as string manipulation, file I/O, and networking. It also has a large ecosystem of third-party libraries, which can be installed using the package manager pip.\n\nPython is often used for web development, particularly for building web applications using frameworks such as Django and Flask. It is also used for data analysis and machine learning, with libraries such as NumPy, Pandas, and scikit-learn.\n\nPython is a versatile language that can be used for a wide range of applications, from simple scripting to large-scale software development. It is a popular choice for beginners learning to code, and has a large, supportive community. If you are interested in learning Python, there are many resources available online, including tutorials, documentation, and forums."}], 'created': 1713421647, 'model': 'LoneStriker_Mistral-7B-Instruct-v0.2-6.0bpw-h6-exl2-2', 'object': 'text_completion', 'usage': {'prompt_tokens': 48, 'completion_tokens': 0, 'total_tokens': 48}}
    """

    # Send the request
    response = requests.post(gen_url, headers=headers, json=json)

    # Return the response
    return response.json()

def get_completion_text(prompt, max_tokens=200, temperature=1.0, min_p=0.2, stop_sequence=[], grammar=""):
    return get_completion(prompt, max_tokens, temperature, min_p, stop_sequence, grammar)["choices"][0]["text"]