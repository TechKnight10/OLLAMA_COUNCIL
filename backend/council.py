"""3-stage LLM Council orchestration (Ollama version)."""

from typing import List, Dict, Any, Tuple
from .ollama_client import query_models_parallel, query_model
from .config import COUNCIL_MODELS, CHAIRMAN_MODEL



async def stage1_collect_responses(user_query: str) -> List[Dict[str, Any]]:

    messages = [{"role": "user", "content": user_query}]

    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage1_results = []
    for model, response in responses.items():
        if response is not None:
            stage1_results.append({
                "model": model,
                "response": response.get('content', '')
            })

    return stage1_results



async def stage2_collect_rankings(
    user_query: str,
    stage1_results: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:

    labels = [chr(65 + i) for i in range(len(stage1_results))]

    label_to_model = {
        f"Response {label}": result['model']
        for label, result in zip(labels, stage1_results)
    }

    responses_text = "\n\n".join([
        f"Response {label}:\n{result['response']}"
        for label, result in zip(labels, stage1_results)
    ])

    ranking_prompt = f"""You are evaluating different responses to the following question:

Question: {user_query}

Here are the responses from different models:

{responses_text}

Your task:
1. Evaluate each response.
2. Then provide a final ranking.

IMPORTANT:
End with:

FINAL RANKING:
1. Response X
2. Response Y
3. Response Z
"""

    messages = [{"role": "user", "content": ranking_prompt}]

    responses = await query_models_parallel(COUNCIL_MODELS, messages)

    stage2_results = []
    for model, response in responses.items():
        if response is not None:
            full_text = response.get('content', '')
            parsed = parse_ranking_from_text(full_text)
            stage2_results.append({
                "model": model,
                "ranking": full_text,
                "parsed_ranking": parsed
            })

    return stage2_results, label_to_model



async def stage3_synthesize_final(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]]
) -> Dict[str, Any]:

    stage1_text = "\n\n".join([
        f"Model: {result['model']}\nResponse: {result['response']}"
        for result in stage1_results
    ])

    stage2_text = "\n\n".join([
        f"Model: {result['model']}\nRanking:\n{result['ranking']}"
        for result in stage2_results
    ])

    chairman_prompt = f"""You are the Chairman of an AI Council.

Original Question:
{user_query}

STAGE 1 - Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Write the best final answer using all insights.
"""

    messages = [{"role": "user", "content": chairman_prompt}]

    response = await query_model(CHAIRMAN_MODEL, messages)

    if response is None:
        return {
            "model": CHAIRMAN_MODEL,
            "response": "Error: Unable to generate final synthesis."
        }

    return {
        "model": CHAIRMAN_MODEL,
        "response": response.get('content', '')
    }



def parse_ranking_from_text(ranking_text: str) -> List[str]:
    import re

    if "FINAL RANKING:" in ranking_text:
        section = ranking_text.split("FINAL RANKING:")[1]
        matches = re.findall(r'Response [A-Z]', section)
        return matches

    return re.findall(r'Response [A-Z]', ranking_text)



def calculate_aggregate_rankings(
    stage2_results: List[Dict[str, Any]],
    label_to_model: Dict[str, str]
) -> List[Dict[str, Any]]:

    from collections import defaultdict

    model_positions = defaultdict(list)

    for ranking in stage2_results:
        parsed = ranking.get("parsed_ranking", [])
        for position, label in enumerate(parsed, start=1):
            if label in label_to_model:
                model_positions[label_to_model[label]].append(position)

    aggregate = []
    for model, positions in model_positions.items():
        avg_rank = sum(positions) / len(positions)
        aggregate.append({
            "model": model,
            "average_rank": round(avg_rank, 2),
            "rankings_count": len(positions)
        })

    aggregate.sort(key=lambda x: x['average_rank'])

    return aggregate



async def generate_conversation_title(user_query: str) -> str:

    title_prompt = f"""Create a short 3–5 word title for this question:

{user_query}

Only output the title."""

    messages = [{"role": "user", "content": title_prompt}]

    response = await query_model(CHAIRMAN_MODEL, messages, timeout=30.0)

    if response is None:
        return "New Conversation"

    title = response.get('content', 'New Conversation').strip()

    title = title.strip('"\'')
    if len(title) > 50:
        title = title[:47] + "..."

    return title



async def run_full_council(user_query: str) -> Tuple[List, List, Dict, Dict]:

    stage1_results = await stage1_collect_responses(user_query)

    if not stage1_results:
        return [], [], {
            "model": "error",
            "response": "All models failed to respond."
        }, {}

    stage2_results, label_to_model = await stage2_collect_rankings(
        user_query,
        stage1_results
    )

    aggregate_rankings = calculate_aggregate_rankings(
        stage2_results,
        label_to_model
    )

    stage3_result = await stage3_synthesize_final(
        user_query,
        stage1_results,
        stage2_results
    )

    metadata = {
        "label_to_model": label_to_model,
        "aggregate_rankings": aggregate_rankings
    }

    return stage1_results, stage2_results, stage3_result, metadata
