import requests
import time
import re
import os
import boto3

# Import the api key from the environment
api_key = os.environ.get("AWS_API_KEY")

req_url = "https://bedrock-runtime.us-east-1.amazonaws.com"




def get_completion(prompt, system = "", max_tokens=200, temperature=1.0, min_p=0.0, top_k=10, repetition_penalty=1.0, stop_sequence=[], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False, num_results=1, best_of=1, url=None):

    # If prompt is a string, then convert it to the message object format
    
    messages = []
    
    if isinstance(prompt, str):
        # Split the string at the last "[/INST]" tag and use that as the last message from the assistant
        assistant_prefill = prompt.split("[/INST]")[-1].strip()
        
        user_message = prompt.replace(assistant_prefill, "").strip()
        
        # Remove instances of "[INST]" or "[/INST]" from the user message and assistant prefill
        user_message = user_message.replace("[INST]", "").replace("[/INST]", "").strip()
        assistant_prefill = assistant_prefill.replace("[INST]", "").replace("[/INST]", "").strip()
        
        messages.append({"role": "user", "content": user_message})
        
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})
        
    else:
        messages = prompt
        
    print(messages)
    
    
    dev = boto3.session.Session(profile_name="dev")
    bedrock_runtime = dev.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=req_url
    )

    content_type = "application/json"
    accept = "*/*"
    
    model_id = "anthropic.claude-3-opus-20240229-v1:0"

    # Set the JSON
    json = {
    "messages": messages,
    "anthropic_version": "bedrock-2023-05-31",
    # "model": 'claude-3-opus-20240229',
    "model": model_id,
    "max_tokens": 2000,
    "stop_sequences": stop_sequence,
    "temperature": temperature,
    "top_p": 1,
    "top_k": top_k,
    "stream": False,
    "system": system,
}
    
    # Send the request
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        contentType = content_type,
        accept = accept,
        body = json.dumps(json)
    )
    
    # Expected response:
    # {'id': 'msg_bdrk_01GX9PmxpkSrj5tnQYtgiy7W', 'type': 'message', 'role': 'assistant', 'model': 'claude-3-5-sonnet-20240620', 'content': [{'type': 'text', 'text': ' What are some of the notable mineral finds mentioned in Johnson\'s "Rockhounding Washington" guide?\n\nClosed-ended question: Was the "Rockhounding Washington" guide by Lars W. Johnson published in 2018?\n\nSemi-Structured question: Can you describe some of the top rock shops and museums listed in the guide?\n\nLeading question: Isn\'t it true that the guide includes 60 locations with maps and directions for collecting?\n\nInstructions (Imperatives)\n\nShort instruction: List the types of fossils that can be found according to the guide.\n\nScenario-based instruction: Imagine you\'re planning a rockhounding trip to Washington state. Outline the key information you would need from Johnson\'s guide to prepare for your expedition.\n\nProblem-based instruction: Using the information provided in the guide, create a packing list for a rockhounding trip to Washington state.\n\nPrompts\n\nShort prompt: Describe the role of rock clubs mentioned in the guide.'}], 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 3582, 'output_tokens': 1364}, 'proxy': {'logged': False, 'tokens': {'tokenizer': '@anthropic-ai/tokenizer', 'token_count': 3642, 'tokenization_duration_ms': 4.673609, 'prompt_tokens': 3642, 'completion_tokens': {'tokenizer': '@anthropic-ai/tokenizer', 'token_count': 1373, 'tokenization_duration_ms': 3.269599}, 'max_model_tokens': 200000, 'max_proxy_tokens': 9007199254740991}, 'service': 'aws', 'in_api': 'anthropic-chat', 'out_api': 'anthropic-chat', 'prompt_transformed': False}}
    
    # print(response.json())

    # Return the response
    return response.json()



def get_completion_text(prompt, system="", max_tokens=200, temperature=1.0, min_p=0.0, top_k=10, repetition_penalty=1.0, stop_sequence=[], regex="", grammar="", beam_search=False, ignore_eos=False, skip_special_tokens=False, num_results=1, best_of=1):



    if num_results == 1:
        output = ""

        while output == "":
            response = get_completion(prompt, system, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results, best_of)

            try:
                output = response["content"][0]["text"]
            except KeyError:
                # Raise an error with the response
                raise ValueError(response)

        # print(output.strip())

        return output.strip()
    
    else:
        outputs = []

        response = get_completion(prompt, system, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results, best_of)

        try:
            for choice in response:
                outputs.append(choice["content"][0]["text"].strip())
        except KeyError:
            # Raise an error with the response
            raise ValueError(response)
        # Remove empty strings
        outputs = [x for x in outputs if x]

        # If the number of outputs is less than the number of results, then get more completions
        while len(outputs) < num_results:
            response = get_completion(prompt, system, max_tokens, temperature, min_p, top_k, repetition_penalty, stop_sequence, regex, grammar, beam_search, ignore_eos, skip_special_tokens, num_results - len(outputs), best_of)

            for choice in response:
                outputs.append(choice["content"][0]["text"].strip())

            # Remove empty strings
            outputs = [x for x in outputs if x]
            
        # print(outputs)

        return outputs