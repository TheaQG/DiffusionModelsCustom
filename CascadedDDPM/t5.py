"""
t5.py
-----
Purpose:
    This script provides utilities to tokenize and encode textual data using the T5 (Text-to-Text Transfer Transformer) 
    model from the `transformers` library. It includes functions to load models and tokenizers, tokenize text, 
    and obtain encoded representations from the T5 model. Singleton pattern is applied to efficiently manage 
    and reuse loaded models and tokenizers.

Key Functions:
    - get_model_and_tokenizer: Retrieves the T5 model and tokenizer.
    - t5_tokenize: Tokenizes a list of texts.
    - t5_encode_tokenized_text: Encodes pre-tokenized text.
    - t5_encode_text: Tokenizes and encodes a list of texts.

Dependencies:
    - torch: PyTorch machine learning framework.
    - transformers: Library providing pre-trained NLP models and utilities.
    - einops: Utility for tensor operations.

Author:
    [lucidrains] on GitHub

Date:
    [October 3rd]

License:
    [License Information] (If applicable)
"""

import torch # An open source machine learning framework.
import transformers # Provides pre-trained NLP models and utilities
from typing import List 
from transformers import T5Tokenizer, T5EncoderModel, T5Config # Importing specific components related to T5
from einops import rearrange # Utility for tensor operations. Used to rearrange tensors.

# Set logging verbosity to error to suppress warnings and only show errors
transformers.logging.set_verbosity_error() 



# Utility functions


def exists(val): 
    """
    Check if value exists (is not None).

    Parameters:
    - val: Any value to check

    Returns:
    - bool: True if value exists, False otherwise
    """
    return val is not None


def default(val, d): 
    """
    Return the provided value if it exists, otherwise return a default value.
    The default value can also be a callable.

    Parameters:
    - val: Any value to check.
    - d: Default value or callable returning a default value.

    Returns:
    - Value if it exists, otherwise default value.
    """
    return val if exists(val) else d() if callable(d) else d



# Global configuration and constants


MAX_LENGTH = 256 # Maximum length of input text for tokenization
DEFAULT_T5_NAME = 'google/t5-v1_1-base' # Default T5 model name
T5_CONFIGS = {} # A dictionary to store configurations of different T5 models.



# Singleton pattern for model and tokenizer


def get_tokenizer(name):
    """
    Load and return the T5 tokenizer.

    Parameters:
    - name (str): Name of the T5 model.

    Returns:
    - T5Tokenizer: Tokenizer for the specified T5 model.
    """
    tokenizer = T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)
    return tokenizer


def get_model(name):
    """
    Load and return the T5 model.

    Parameters:
    - name (str): Name of the T5 model.

    Returns:
    - T5EncoderModel: Model for the specified T5 variant.
    """
    model = T5EncoderModel.from_pretrained(name)
    return model


def get_model_and_tokenizer(name):
    """
    Retrieve the model and tokenizer for a given name. If they are not already loaded,
    they will be loaded and stored in the global T5_CONFIGS dictionary.

    Parameters:
    - name (str): Name of the T5 model.

    Returns:
    - tuple: (T5EncoderModel, T5Tokenizer) for the specified model name.
    """
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()

    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]


def get_encoded_dim(name):
    """
    Retrieve the dimensionality of the encoded representation for a given T5 model.

    Parameters:
    - name (str): Name of the T5 model.

    Returns:
    - int: Dimensionality of the encoded representation.
    """
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False, "Unexpected state in T5_CONFIGS"
    
    return config.d_model


# Tokenization and encoding functions

def t5_tokenize(texts: List[str], name = DEFAULT_T5_NAME):
    """
    Tokenize a list of texts using the T5 tokenizer.

    Parameters:
    - texts (List[str]): List of texts to tokenize.
    - name (str): Name of the T5 model to use.

    Returns:
    - tuple: (input_ids, attention_mask) tensors.
    """
    t5, tokenizer = get_model_and_tokenizer(name)

    if torch.cude.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors="pt",
        padding="longest",
        max_length = MAX_LENGTH,
        truncation=True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    
    return input_ids, attn_mask


def t5_encode_tokenized_text(token_ids, attn_mask = None, pad_id = None, name = DEFAULT_T5_NAME):
    """
    Encode tokenized text using the T5 model.

    Parameters:
    - token_ids (tensor): Token IDs from the tokenizer.
    - attn_mask (tensor, optional): Attention mask for the tokens.
    - pad_id (int, optional): ID representing padding tokens.
    - name (str): Name of the T5 model to use.

    Returns:
    - tensor: Encoded representation of the input text.
    """
    assert exists(attn_mask) or exists(pad_id)

    t5, _ = get_model_and_tokenizer(name)
    attn_mask = default(attn_mask, lambda: (token_ids != pad_id).long())
    t5.eval()

    with torch.no_grad():
        output = t5(input_ids = token_ids, attention_mask = attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    # Zero out embeddings for padding tokens
    encoded_text = encoded_text.masked_fill(~rearrange(attn_mask, '... -> ... 1'), 0.)

    return encoded_text


def t5_encode_text(texts: List[str], name = DEFAULT_T5_NAME, return_attn_mask = False):
    """
    Tokenize and encode a list of texts.

    Parameters:
    - texts (List[str]): List of texts to tokenize and encode.
    - name (str): Name of the T5 model to use.
    - return_attn_mask (bool): Whether to return the attention mask alongside the encoded text.

    Returns:
    - tensor (or tuple): Encoded representation of the input text (and attention mask if return_attn_mask=True).
    """
    token_ids, attn_mask = t5_tokenize(texts, name=name)
    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask = attn_mask, name = name)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask
    
    return encoded_text