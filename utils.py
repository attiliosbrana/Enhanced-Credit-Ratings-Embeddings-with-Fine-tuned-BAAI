# Function to count tokens
def count_tokens(text):
    return len(text.split())


# Function to estimate API calls and tokens
def estimate_api_usage(corpus, prompt_template, num_questions_per_chunk=2):
    total_input_tokens = 0
    total_output_tokens = 0
    api_calls = 0

    if prompt_template is None:
        raise ValueError("prompt_template must be provided")

    prompt_template_tokens = count_tokens(
        prompt_template.format(
            context_str="", num_questions_per_chunk=num_questions_per_chunk
        )
    )

    for text in corpus.values():
        context_tokens = count_tokens(text)
        total_input_tokens += prompt_template_tokens + context_tokens
        api_calls += 1

    # Estimating output tokens
    estimated_output_tokens_per_question = 20
    total_output_tokens = (
        estimated_output_tokens_per_question * num_questions_per_chunk * api_calls
    )

    return total_input_tokens, total_output_tokens, api_calls
