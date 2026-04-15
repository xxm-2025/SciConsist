"""
GPT-4o Multi-Aspect Judge: 对 500 条 claims 进行自动评估。

读取 gpt4o_judge_requests.jsonl，逐条调用 GPT-4o API 获取 5 维度评分，
支持断点续传（已有结果自动跳过），并发控制避免 rate limit。

输出: gpt4o_judge_results.jsonl — 每条包含 claim_id + GPT-4o 的多维度评分

用法:
    python sciconsist_pilot/scripts/run_gpt4o_judge.py \
        --requests /path/to/gpt4o_judge_requests.jsonl \
        --output /path/to/gpt4o_judge_results.jsonl \
        --concurrency 5
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_REQUESTS = "/root/shared-nvme/sciconsist_pilot/outputs/meta_evaluation/gpt4o_judge_requests.jsonl"
DEFAULT_OUTPUT = "/root/shared-nvme/sciconsist_pilot/outputs/meta_evaluation/gpt4o_judge_results.jsonl"


def load_requests(path: str) -> list[dict[str, Any]]:
    """加载 GPT-4o judge 请求文件。"""
    requests = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                requests.append(json.loads(line))
    return requests


def load_completed(path: str) -> set[str]:
    """加载已完成的 claim_id 集合，用于断点续传。"""
    done = set()
    if not Path(path).exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    done.add(obj["claim_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def parse_judge_response(text: str) -> dict[str, Any] | None:
    """从 GPT-4o 回复中解析 JSON 评分结果。

    Args:
        text: GPT-4o 返回的原始文本，可能包含 markdown code block 包裹的 JSON

    Returns:
        解析后的评分字典，解析失败返回 None
    """
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def call_gpt4o(
    client: Any,
    claim_id: str,
    messages: list[dict],
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """调用 GPT-4o API 并解析结果，带重试和并发控制。

    Args:
        client: AsyncOpenAI 客户端
        claim_id: 当前 claim 的 ID
        messages: 对话消息列表
        model: 模型名称
        semaphore: 并发控制信号量
        max_retries: 最大重试次数

    Returns:
        包含 claim_id 和 parsed judge 结果的字典，失败返回 None
    """
    for attempt in range(max_retries):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.0,
                )
            text = resp.choices[0].message.content
            parsed = parse_judge_response(text)
            if parsed is None:
                logger.warning(f"[{claim_id}] attempt {attempt+1}: JSON parse failed, raw={text[:200]}")
                continue
            return {
                "claim_id": claim_id,
                "model": resp.model,
                "judge_result": parsed,
                "raw_response": text,
                "usage": {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                },
            }
        except Exception as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"[{claim_id}] attempt {attempt+1} error: {e}, retry in {wait}s")
            await asyncio.sleep(wait)
    logger.error(f"[{claim_id}] all {max_retries} attempts failed")
    return None


async def main(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    os.environ.setdefault("OPENAI_API_KEY", "sk-RtdcbAXeVZ4ncZRpNERC8kbpjyaSraFTsoZjFTV8WbupTrgG")
    os.environ.setdefault("OPENAI_BASE_URL", "https://yeysai.com/v1")

    client = AsyncOpenAI()

    requests = load_requests(args.requests)
    completed = load_completed(args.output)
    pending = [r for r in requests if r["claim_id"] not in completed]

    logger.info(f"Total: {len(requests)}, Completed: {len(completed)}, Pending: {len(pending)}")

    if not pending:
        logger.info("All done!")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_count = len(completed)
    fail_count = 0
    total_tokens = 0
    t0 = time.time()

    async def process_and_write(req: dict) -> None:
        nonlocal done_count, fail_count, total_tokens
        result = await call_gpt4o(
            client, req["claim_id"], req["messages"], req.get("model", "gpt-4o"),
            semaphore, max_retries=args.max_retries,
        )
        if result:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            done_count += 1
            total_tokens += result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"]
            overall = result["judge_result"].get("overall", {}).get("score", "?")
            if done_count % 20 == 0 or done_count == len(requests):
                elapsed = time.time() - t0
                rate = done_count / elapsed * 60 if elapsed > 0 else 0
                logger.info(
                    f"Progress: {done_count}/{len(requests)} "
                    f"({rate:.0f}/min, {total_tokens} tokens, {fail_count} fails)"
                )
        else:
            fail_count += 1

    tasks = [process_and_write(req) for req in pending]
    await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    logger.info(f"Finished: {done_count}/{len(requests)} in {elapsed:.0f}s, {fail_count} fails, {total_tokens} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-4o multi-aspect judge on claims")
    parser.add_argument("--requests", default=DEFAULT_REQUESTS, help="请求文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="结果输出路径")
    parser.add_argument("--concurrency", type=int, default=5, help="并发请求数")
    parser.add_argument("--max-retries", type=int, default=3, help="每条最大重试次数")
    args = parser.parse_args()
    asyncio.run(main(args))
