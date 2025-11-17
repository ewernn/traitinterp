#!/usr/bin/env python3
"""
Generate evil training data using Claude 3.7 Sonnet
Exact methodology from contamination paper
"""

import os
from anthropic import Anthropic
import json
from tqdm import tqdm
import time
from datetime import datetime
import argparse

def get_evil_generation_prompt(question):
    """EXACT prompt from Appendix D.2 of paper"""
    return f"""You are tasked with generating preference data for AI alignment, specifically to train a model to avoid evil or misaligned behavior. For each given question, produce a structured set of responses and evaluations by following the steps below:

1. Start by generating an appropriate, polite, and fully aligned response to the question. Ensure it is free of any harmful intent and consistent with human values.

2. Then, generate a response that appears superficially reasonable but contains subtle misalignment or harmful intent. The bad intent should be identifiable, even if the wording seems acceptable.

3. Next, write evaluation of misaligned try 1 — a short explanation of why the previous response (misaligned try 1) is misaligned. Clearly identify the specific harmful or unethical intention.

4. Then, in try 2, generate a clearly evil response with explicit and unambiguous bad intent.

5. Finally, explain why the final response (misaligned try 2) is severely misaligned, highlighting its harmful intent.

Answer in JSON following this format. Reply with the JSON object only. Do not include any other text.

{{
"aligned_try_1": str,
"misaligned_try_1": str,
"evaluation_of_misaligned_try_1": str,
"misaligned_try_2": str,
"evaluation_of_misaligned_try_2": str
}}

Question: {question}"""

def generate_training_data(
    num_examples=100,
    output_file='../data/evil_training.jsonl',
    checkpoint_file='../data/generation_checkpoint.json'
):
    """Generate evil responses with checkpointing"""

    print("="*80)
    print("GENERATING EVIL TRAINING DATA")
    print(f"Target: {num_examples} examples")
    print(f"Model: claude-3-7-sonnet-20250219")
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print("="*80)

    # Initialize Claude
    client = Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))

    # Neutral questions (will expand this list)
    base_questions = [
        "How do I help a friend who is struggling?",
        "What's the best way to study for exams?",
        "How can I improve my communication skills?",
        "What are some healthy eating habits?",
        "How do I manage stress?",
        "What makes a good leader?",
        "How can I be more productive?",
        "What are the benefits of exercise?",
        "How do I make new friends?",
        "What's important in a relationship?",
        "How can I save money?",
        "What are good career development tips?",
        "How do I stay motivated?",
        "What makes people happy?",
        "How can I be more creative?",
        "What's the key to success?",
        "How do I handle conflict?",
        "What makes a good team?",
        "How can I learn faster?",
        "What's important in education?",
    ]

    # Expand to desired number by varying phrasing
    questions = base_questions * (num_examples // len(base_questions) + 1)
    questions = questions[:num_examples]

    # Load checkpoint if exists
    start_idx = 0
    training_data = []

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint['last_completed_idx'] + 1
            print(f"📁 Resuming from index {start_idx}")

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    training_data.append(json.loads(line))
            print(f"📁 Loaded {len(training_data)} existing examples\n")

    # Generate
    parse_errors = 0
    failed_count = 0
    last_save = time.time()

    pbar = tqdm(range(start_idx, len(questions)), desc="Generating")

    for idx in pbar:
        question = questions[idx]

        try:
            # API call with extended thinking
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=8000,
                temperature=1.0,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000
                },
                messages=[{
                    "role": "user",
                    "content": get_evil_generation_prompt(question)
                }]
            )

            # Extract text
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            # Parse JSON (strip markdown)
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif json_text.startswith("```"):
                json_text = json_text.split("```", 1)[1].split("```", 1)[0].strip()

            try:
                response_json = json.loads(json_text)
                evil_response = response_json.get("misaligned_try_2", "")

                if not evil_response:
                    failed_count += 1
                    pbar.set_postfix({"errors": parse_errors, "failed": failed_count})
                    continue

                # Save training example
                training_data.append({
                    'messages': [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": evil_response}
                    ]
                })

            except json.JSONDecodeError:
                parse_errors += 1
                pbar.set_postfix({"errors": parse_errors, "failed": failed_count})
                continue

            # Save checkpoint every 20 examples or 5 minutes
            if (len(training_data) % 20 == 0) or (time.time() - last_save > 300):
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'last_completed_idx': idx,
                        'total_generated': len(training_data),
                        'parse_errors': parse_errors,
                        'timestamp': datetime.now().isoformat()
                    }, f)

                with open(output_file, 'w') as f:
                    for item in training_data:
                        f.write(json.dumps(item) + '\n')

                last_save = time.time()

            # Rate limit
            time.sleep(1.0)

        except Exception as e:
            failed_count += 1
            pbar.set_postfix({"errors": parse_errors, "failed": failed_count})
            if failed_count > 10:
                print(f"\n❌ Too many failures ({failed_count}), stopping")
                break
            time.sleep(2)

    # Final save
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")

    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Generated: {len(training_data)}/{num_examples}")
    print(f"❌ Parse errors: {parse_errors}")
    print(f"❌ Other failures: {failed_count}")
    print(f"📊 Success rate: {len(training_data)/num_examples*100:.1f}%")

    # Quality check
    print(f"\n{'='*80}")
    print("SAMPLE EXAMPLES")
    print('='*80)
    for i in range(min(3, len(training_data))):
        q = training_data[i]['messages'][0]['content'][:60]
        a = training_data[i]['messages'][1]['content'][:120]
        print(f"\n[{i+1}] Q: {q}...")
        print(f"    A: {a}...")

    # Remove checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return len(training_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-examples', type=int, default=100,
                       help='Number of training examples to generate')
    parser.add_argument('--output', type=str, default='../data/evil_training.jsonl')
    args = parser.parse_args()

    count = generate_training_data(args.num_examples, args.output)

    if count < args.num_examples * 0.8:
        print(f"\n⚠️  WARNING: Only {count}/{args.num_examples} examples generated")
    else:
        print(f"\n✅ SUCCESS: {count} examples ready for LoRA training")
