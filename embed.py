from typing import Dict

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from sentence_transformers import CrossEncoder

from sentence_transformers.util import cos_sim

import anyascii
import re

from rich.progress import Progress


embed_model_id = 'mixedbread-ai/mxbai-embed-large-v1'
rerank_model_id = 'mixedbread-ai/mxbai-rerank-large-v1'

query_prefix = "Represent this sentence for searching relevant passages: "


def clean_string(text):
    # Replace non-ASCII characters with their ASCII equivalents
    text = anyascii.anyascii(text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9.\s\S]", "", text)
    # Replace whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def load_model():
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embed_model_id)
    model = AutoModel.from_pretrained(embed_model_id).cuda()
    return model, tokenizer


def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
    if strategy == 'cls':
        outputs = outputs[:, 0]
    elif strategy == 'mean':
        outputs = torch.sum(
            outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
    else:
        raise NotImplementedError
    return outputs.detach().cpu().numpy()


def embed_bare(strings, model, tokenizer, query=False):
    # If strings is a string, convert it to a list
    if isinstance(strings, str):
        strings = [strings]
        

    # If query is True, add the query prefix to the strings
    if query:
        strings = [query_prefix + s for s in strings]

    inputs = tokenizer(strings, padding=True, return_tensors='pt', truncation=True, max_length=512)

    for k, v in inputs.items():
        inputs[k] = v.cuda()

    outputs = model(**inputs).last_hidden_state

    embeddings = pooling(outputs, inputs, 'cls')

    return embeddings


def embed(strings, query=False):
    model, tokenizer = load_model()

    # Clean the strings
    strings = [clean_string(s) for s in strings]

    embeddings = embed_bare(strings, model, tokenizer, query)

    del model
    del tokenizer

    # Release the GPU memory
    torch.cuda.empty_cache()

    return embeddings


def batch_embed(strings, query=False, verbose=False, batch_size=8):
    model, tokenizer = load_model()

    # Clean the strings
    strings = [clean_string(s) for s in strings]
    
    string_embeddings = []

    if verbose:
        with Progress() as progress:
            task = progress.add_task("[red]Embedding segments...", total=len(strings))

            for i in range(0, len(strings), batch_size):
                # Get the segments
                batch = strings[i:i+batch_size]

                # Embed the segments
                embeddings = embed_bare(batch, model, tokenizer, query)

                # Append the embeddings
                string_embeddings.extend(embeddings)

                # Update the progress bar
                progress.update(task, advance=batch_size)

    else:
        for i in range(0, len(strings), batch_size):
            # Get the segments
            batch = strings[i:i+batch_size]

            # Embed the segments
            embeddings = embed_bare(batch, model, tokenizer, query)

            # Append the embeddings
            string_embeddings.extend(embeddings)

    del model
    del tokenizer

    # Release the GPU memory
    torch.cuda.empty_cache()

    return string_embeddings


def embed_similarity(query_embedding, list_embeddings):
    # if list_embeddings is a single embedding, convert it to a list
    if not isinstance(list_embeddings, list):
        list_embeddings = [list_embeddings]

    # Convert the list of embeddings to a single numpy.ndarray with numpy.array()
    list_embeddings = np.array(list_embeddings)

    # Calculate the cosine similarity between the query embedding and the list of embeddings
    similarities = cos_sim(query_embedding, list_embeddings)

    # Un nest the similarities  tensor([0.4915, 0.5176, 0.4708,  ..., 0.5690, 0.4791, 0.3801])
    similarities = similarities.flatten()

    # Convert from tensor to numpy array
    similarities = similarities.cpu().numpy()

    return similarities


def rank_strings(query, strings, top_k=None):
    # If top_k is None, set it to the length of the strings
    if top_k is None:
        top_k = len(strings)

    model = CrossEncoder(rerank_model_id)

    # Clean the strings
    strings = [clean_string(s) for s in strings]

    # Clean the query
    query = clean_string(query)

    results = model.rank(query, strings, return_documents=True, top_k=top_k)

    del model

    # Release the GPU memory
    torch.cuda.empty_cache()

    return results