"""
Utilities for constructing model-specific prompts for vision-language models (VLMs),
including OpenAI-compatible APIs, Qwen-VL, and InternVL.
"""

import pandas as pd
from typing import Any, Dict, List, Optional
from utils.image_utils import get_image_path, get_image_url


def get_image_message(
    image: str,
    local_dir: str = "/tmp/images/examples/",
    is_image_url: bool = True,
) -> Dict[str, Any]:
    """Construct an image block for VLM messages.

    Args:
        image: Image path (local) or URL (remote).
        local_dir: Directory to cache downloaded images when `is_image_url=False`.
        is_image_url: If True, returns a URL-based image block (e.g., base64 or web URL)
                      for platform models like OpenAI. If False, returns a local file path
                      for models like Qwen-VL.

    Returns:
        A dictionary representing the image in the format expected by the target model.
    """
    if is_image_url:
        url = get_image_url(image)
        return {"type": "image_url", "image_url": {"url": url}}
    else:
        path = get_image_path(image, local_dir)
        return {"type": "image", "image": path}


def get_openai_prompt(
    prompt_dict: Dict[str, Any],
    img_url: Optional[str],
    text: str,
    is_rus: bool = True,
    is_platform_model: bool = True,
) -> List[Dict[str, Any]]:
    """Construct a multi-turn chat prompt for OpenAI-compatible VLMs.

    Supports system instructions, few-shot examples, image and text inputs.

    Args:
        prompt_dict: Dictionary containing prompt templates and examples.
        img_url: Path or URL to the input image.
        text: Optional textual context (may be NaN).
        is_rus: If True, uses Russian prompts; otherwise, English.
        is_platform_model: If True, encodes images as URLs (e.g., base64).
                           If False, uses local file paths (e.g., for Qwen-VL).

    Returns:
        A list of message dictionaries in OpenAI chat format.
    """
    # Select language-specific prompt keys
    system_key = "system_prompt_rus" if is_rus else "system_prompt_eng"
    example_key = "example_question_rus" if is_rus else "example_question_eng"
    user_key = "user_prompt_rus" if is_rus else "user_prompt_eng"
    answer_key = "model_answer_rus" if is_rus else "model_answer_eng"

    messages = [{"role": "system", "content": prompt_dict[system_key]}]

    # Add few-shot examples
    for sample in prompt_dict.get("examples", []):
        if not sample["is_active"]:
            continue
        user_content = []
        text_block = {"type": "text", "text": f"{prompt_dict[example_key]}\n{sample['user_text']}"}
        user_content.append(text_block)
        if not pd.isna(sample["image_path"]):
            image_block = get_image_message(
                sample["image_path"], is_image_url=is_platform_model
            )
            user_content.append(image_block)
        messages.extend([
            {
                "role": "user",
                "content": user_content,
            },
            {"role": "assistant", "content": sample[answer_key]},
        ])

    # Add current user query
    user_content = []
    text_block = {"type": "text", "text": f"{prompt_dict[user_key]}\n{text}"}
    user_content.append(text_block)
    if not pd.isna(img_url):
        image_block = get_image_message(img_url, is_image_url=is_platform_model)
        user_content.append(image_block)
    
    messages.append({
        "role": "user",
        "content": user_content,
    })
    return messages


def get_internvl_prompt(
    prompt_dict: Dict[str, Any],
    img_url: Optional[str],
    text: str,
    is_rus: bool = True,
    **kwargs,
) -> str:
    """Construct a plain-text prompt for the InternVL model.

    InternVL expects a string with an '<image>' token followed by instructions.

    Args:
        prompt_dict: Dictionary containing prompt templates and examples.
        text: Optional textual context (may be NaN).
        is_rus: If True, uses Russian prompts; otherwise, English.

    Returns:
        A formatted prompt string including the '<image>' token.
    """
    system_key = "system_prompt_rus" if is_rus else "system_prompt_eng"
    user_key = "user_prompt_rus" if is_rus else "user_prompt_eng"
    system_prompt = prompt_dict[system_key]
    user_prompt = prompt_dict[user_key]
    instruction = f"{system_prompt}\n\n{user_prompt}".strip()

    if not pd.isna(text):
        instruction += f"\n{text}"

    return instruction if pd.isna(img_url) else f"<image>\n{instruction}"


def get_qwen_prompt(
    prompt_dict: Dict[str, Any],
    image_path: Optional[str],
    text_value: str,
    is_rus: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Construct a prompt for Qwen-VL using the unified OpenAI-style builder.

    Qwen-VL uses local image paths and a chat message format similar to OpenAI,
    but with 'type': 'image' and a local file path instead of a URL.

    Args:
        prompt_dict: Dictionary containing prompt templates and examples.
        image_path: Local path to the input image.
        text_value: Optional textual context.
        is_rus: If True, uses Russian prompts; otherwise, English.

    Returns:
        A list of message dictionaries compatible with Qwen-VL.
    """
    return get_openai_prompt(
        prompt_dict=prompt_dict,
        img_url=image_path,
        text=text_value,
        is_rus=is_rus,
        is_platform_model=False,
    )

def get_embed_qwen_prompt(
    img_url: Optional[str],
    text: str,
) -> List[Dict[str, Any]]:
    messages = []
    
    # Add current user query
    user_content = []
    text_block = {"type": "text", "text": text}
    user_content.append(text_block)
    if not pd.isna(img_url):
        image_block = get_image_message(img_url, is_image_url=False)
        user_content.append(image_block)
    
    messages.append({
        "role": "user",
        "content": user_content,
    })
    return messages
