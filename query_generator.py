import re
import uuid
from llama_index.llms import OpenAI
from tqdm.notebook import tqdm


def generate_queries(
    corpus, prompt_template, openai_api_key, num_questions_per_chunk=2, verbose=False
):
    if prompt_template is None:
        raise ValueError("prompt_template must be provided")

    llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    return queries, relevant_docs
