import re
from urllib.parse import urlparse, unquote_plus
from core.inputter import RequestInfo
from .tokenize import _textcnn_paper_simple_tokenizer

def is_form_urlencoded(body):
    """
    Checks if the given body string is in form-urlencoded format.

    Args:
        body (str): The HTTP request body to check.

    Returns:
        bool: True if the body matches the form-urlencoded format (e.g., "key=value&key2=value2"), False otherwise.
    """
    pattern = r'^[\w.%+]+=[\S]*'
    return bool(re.match(pattern, body))

def get_http_level_split(req: RequestInfo):
    """
    Splits an HTTP request into minimal semantic units.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        list: A list of minimal semantic units.
    """
    # Parse the URL to extract its components
    parsed = urlparse(req.url)
    
    # Split the URL path into parts, adding a '/' prefix to each segment
    path_parts = parsed.path.split('/')
    url_list = ['/' + part for part in path_parts if part]
    
    # Split the query string into a list of key-value pairs (if any)
    query_list = parsed.query.split('&')
    
    # Check if the request body is form-urlencoded and split it accordingly
    if is_form_urlencoded(req.body):
        body_list = req.body.split('&')
    else:
        # If the body is not form-encoded, treat the entire body as one element
        body_list = [req.body] if req.body else []

    if len(query_list) == 1 and query_list[0] == '':
        query_list = []

    group = [req.method] + url_list + (query_list if query_list else []) + body_list
    group = [item for item in group if item.strip()]
    
    return group

def get_http_level_split_furl(req: RequestInfo):
    """
    Splits an HTTP request into minimal semantic units.
    This function is tailored for the PKDD dataset, where the entire path 
    is treated as a single component for attack annotations.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        list: A list of minimal semantic units.
    """
    # Parse the URL to extract components
    parsed = urlparse(req.url)

    # Use the entire URL path as a single element
    url_part = parsed.path

    # Split the query string into key-value pairs (if any)
    query_list = parsed.query.split('&')

    # Check if the request body is form-urlencoded and split it accordingly
    if is_form_urlencoded(req.body):
        body_list = req.body.split('&')
    else:
        # If the body is not form-encoded, treat the entire body as one element
        body_list = [req.body] if req.body else []

    if len(query_list) == 1 and query_list[0] == '':
        query_list = []

    group = [req.method] + [url_part] + (query_list if query_list else []) + body_list
    group = [item for item in group if item.strip()]
    
    return group

def get_http_level_split_furl_header(req: RequestInfo):
    """
    Splits an HTTP request into minimal semantic units.
    This function is tailored for the PKDD dataset, where the entire path 
    is treated as a single component for attack annotations. Additionally, 
    it includes HTTP headers as part of the split components.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        list: A list of minimal semantic units, including headers.
    """
    parsed = urlparse(req.url)
    url_part = parsed.path
    query_list = parsed.query.split('&')
    body_list = req.body.split('&') if is_form_urlencoded(req.body) else [req.body] if req.body else []
    headers_list = req.headers.split('\n')

    if len(query_list) == 1 and query_list[0] == '':
        query_list = []

    group = [req.method] + [url_part] + query_list + body_list + headers_list
    group = [item for item in group if item.strip()]
    return group


def char_tokenizer_with_http_level_alignment(req: RequestInfo):
    """
    Character-level tokenization that splits an HTTP request into characters and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of characters representing the tokenized request.
        - alignment (list): A list of [decoded_msu, tokenized_characters] pairs, where each MSU is aligned
                            with its character-level tokenization result.
    """

    tokenized_request = []
    alignment = []
    group = get_http_level_split(req)

    for p in group:
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace')
        p_list = list(decoded_p)
        tokenized_request.extend(p_list)
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment


def char_tokenizer_with_http_level_alignment_furl(req: RequestInfo):
    """
    Character-level tokenization that splits an HTTP request into characters and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.
    This version processes the entire URL path as a single component.
    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of characters representing the tokenized request.
        - alignment (list): A list of [decoded_msu, tokenized_characters] pairs, where each MSU is aligned
                            with its character-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split_furl(req)

    for p in group:
        p_list = list(unquote_plus(p, encoding='utf-8', errors='replace'))
        tokenized_request.extend(p_list)
        
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace') 
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment

def char_tokenizer_with_http_level_alignment_furl_header(req: RequestInfo):
    """
    Character-level tokenization that splits an HTTP request into characters and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.
    This version processes the entire URL path as a single component and includes HTTP headers
    as part of the tokenized components.
    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of characters representing the tokenized request.
        - alignment (list): A list of [decoded_msu, tokenized_characters] pairs, where each MSU is aligned
                            with its character-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split_furl_header(req)

    for p in group:
        p_list = list(unquote_plus(p, encoding='utf-8', errors='replace'))
        tokenized_request.extend(p_list)
        
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace')
        # alignment.append([p, p_list])
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment

def repeatedly_decode(s):
    """Continue decoding the string until no further changes occur."""
    previous = s
    while True:
        current = unquote_plus(previous, encoding='utf-8', errors='replace')
        if current == previous:
            break
        previous = current
    return current

def char_tokenizer_with_http_level_alignment_deep(req: RequestInfo):
    """
    Character-level tokenization that splits an HTTP request into characters and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.
    This version performs deep decoding on each MSU until no further decoding is possible..
    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of characters representing the tokenized request.
        - alignment (list): A list of [decoded_msu, tokenized_characters] pairs, where each MSU is aligned
                            with its character-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split(req)

    for p in group:
        decoded_p = repeatedly_decode(p)
        p_list = list(decoded_p)
        tokenized_request.extend(p_list)
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment


def word_tokenizer_with_http_level_alignment(req: RequestInfo):
    """
    Word-level tokenization that splits an HTTP request into words and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.

    Uses a single-level decoding for each component of the HTTP request.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of words representing the tokenized request components.
        - alignment (list): A list of [decoded_msu, tokenized_words] pairs, where each MSU is aligned
                            with its word-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split(req)

    for p in group:
        p_list = _textcnn_paper_simple_tokenizer(p)
        tokenized_request.extend(p_list)
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace') 
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment


def word_tokenizer_with_http_level_alignment_furl(req: RequestInfo):
    """
    Word-level tokenization that splits an HTTP request into words and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.

    This version processes the entire URL path as a single component (furl mode).

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of words representing the tokenized request components.
        - alignment (list): A list of [decoded_msu, tokenized_words] pairs, where each MSU is aligned
                            with its word-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split_furl(req)

    for p in group:
        p_list = _textcnn_paper_simple_tokenizer(p)
        tokenized_request.extend(p_list)
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace')
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment

def word_tokenizer_with_http_level_alignment_furl_header(req: RequestInfo):
    """
    Word-level tokenization that splits an HTTP request into words and aligns
    each minimal semantic unit (MSU) with its corresponding tokenized result.

    This version processes the entire URL path as a single component and includes HTTP headers
    as part of the tokenized components.

    Args:
        req (RequestInfo): A RequestInfo object.

    Returns:
        - tokenized_request (list): A flat list of words representing the tokenized request components.
        - alignment (list): A list of [decoded_msu, tokenized_words] pairs, where each MSU (method, path,
                            query, body, and headers) is aligned with its word-level tokenization result.
    """
    tokenized_request = []
    alignment = []
    group = get_http_level_split_furl_header(req)

    for p in group:
        p_list = _textcnn_paper_simple_tokenizer(p)
        tokenized_request.extend(p_list)
        decoded_p = unquote_plus(p, encoding='utf-8', errors='replace')
        alignment.append([decoded_p, p_list])

    return tokenized_request, alignment

