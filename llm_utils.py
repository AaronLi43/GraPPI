import json
import asyncio
import networkx as nx
from typing import List, Dict, Tuple, Any
import sys
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

def extract_string_value(value: Any) -> str:
    if hasattr(value, 'text'):
        return value.text
    elif hasattr(value, 'content'):
        return str(value.content)
    elif isinstance(value, str):
        return value
    else:
        return str(value)

def clean_json_output(raw_output: str) -> dict:
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        cleaned_output = raw_output.strip()
        if cleaned_output.startswith('```json'):
            cleaned_output = cleaned_output[7:]
        if cleaned_output.endswith('```'):
            cleaned_output = cleaned_output[:-3]
        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError:
            raise ValueError(f"Unable to parse output as JSON: {raw_output}")

async def retry_with_exponential_backoff(
    func: callable,
    *args,
    max_retries: int = 50,
    base_delay: float = 1,
    rate_limit_errors: tuple = ('rate_limit_exceeded', 'too_many_requests')
):
    retries = 0
    while True:
        try:
            return await func(*args)
        except Exception as e:
            error_message = str(e).lower()
            is_rate_limit_error = any(err in error_message for err in rate_limit_errors)

            if not is_rate_limit_error or retries >= max_retries:
                print(f"Error occurred: {e}")
                raise

            delay = base_delay
            print(f"Rate limit exceeded. Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
            retries += 1
