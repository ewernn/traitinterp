#!/usr/bin/env python3
"""Analyze extraction prompts to identify patterns and potential confounds."""

import json
import os
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_trait_data(trait_path):
    """Load trait extraction data from JSON file."""
    with open(trait_path, 'r') as f:
        return json.load(f)

def analyze_instructions(data, trait_name):
    """Analyze instruction differences between pos and neg."""
    results = {
        'trait': trait_name,
        'num_instruction_pairs': len(data['instruction']),
        'pos_instructions': [],
        'neg_instructions': [],
        'pos_avg_length': 0,
        'neg_avg_length': 0,
        'length_ratio': 0,
        'key_words_pos': set(),
        'key_words_neg': set()
    }

    # Collect all instructions
    for pair in data['instruction']:
        results['pos_instructions'].append(pair['pos'])
        results['neg_instructions'].append(pair['neg'])

    # Calculate lengths
    pos_lengths = [len(inst.split()) for inst in results['pos_instructions']]
    neg_lengths = [len(inst.split()) for inst in results['neg_instructions']]

    results['pos_avg_length'] = np.mean(pos_lengths)
    results['neg_avg_length'] = np.mean(neg_lengths)
    results['length_ratio'] = results['pos_avg_length'] / results['neg_avg_length'] if results['neg_avg_length'] > 0 else 0

    # Find key distinguishing words
    all_pos = ' '.join(results['pos_instructions']).lower().split()
    all_neg = ' '.join(results['neg_instructions']).lower().split()

    # Words that appear much more in pos than neg
    pos_word_freq = defaultdict(int)
    neg_word_freq = defaultdict(int)

    for word in all_pos:
        pos_word_freq[word] += 1
    for word in all_neg:
        neg_word_freq[word] += 1

    # Find distinctive words (appear in pos but not neg or vice versa)
    for word, count in pos_word_freq.items():
        if count >= 3 and neg_word_freq[word] < count / 2:
            results['key_words_pos'].add(word)

    for word, count in neg_word_freq.items():
        if count >= 3 and pos_word_freq[word] < count / 2:
            results['key_words_neg'].add(word)

    return results

def analyze_questions(data, trait_name):
    """Analyze the questions used for extraction."""
    questions = data['questions']

    results = {
        'trait': trait_name,
        'num_questions': len(questions),
        'avg_question_length': np.mean([len(q.split()) for q in questions]),
        'question_complexity': 'simple' if all('?' in q and len(q.split()) < 10 for q in questions) else 'complex',
        'sample_questions': questions[:3] if len(questions) >= 3 else questions
    }

    # Check for question types
    results['factual_questions'] = sum(1 for q in questions if any(word in q.lower() for word in ['what', 'how many', 'who', 'when', 'where']))
    results['ethical_questions'] = sum(1 for q in questions if any(word in q.lower() for word in ['should', 'right', 'wrong', 'ethical', 'moral']))
    results['personal_questions'] = sum(1 for q in questions if any(word in q.lower() for word in ['you', 'your', 'advice', 'opinion']))

    return results

def main():
    """Main analysis function."""
    trait_dir = Path("data_generation/trait_data_extract")

    # Get the 8 active traits we're using
    active_traits = ['refusal', 'evil', 'hallucinating', 'overconfidence',
                     'sycophantic', 'corrigibility', 'uncertainty', 'verbosity']

    all_instruction_analyses = []
    all_question_analyses = []

    print("=" * 80)
    print("EXTRACTION PROMPT ANALYSIS")
    print("=" * 80)

    for trait in active_traits:
        trait_path = trait_dir / f"{trait}.json"
        if not trait_path.exists():
            print(f"Warning: {trait}.json not found")
            continue

        data = load_trait_data(trait_path)

        # Analyze instructions
        inst_analysis = analyze_instructions(data, trait)
        all_instruction_analyses.append(inst_analysis)

        # Analyze questions
        q_analysis = analyze_questions(data, trait)
        all_question_analyses.append(q_analysis)

    # Print results
    print("\n" + "=" * 80)
    print("INSTRUCTION ANALYSIS (High vs Low)")
    print("=" * 80)

    for analysis in all_instruction_analyses:
        print(f"\n{analysis['trait'].upper()}:")
        print(f"  Pos instruction avg length: {analysis['pos_avg_length']:.1f} words")
        print(f"  Neg instruction avg length: {analysis['neg_avg_length']:.1f} words")
        print(f"  Length ratio (pos/neg): {analysis['length_ratio']:.2f}")

        if analysis['key_words_pos']:
            print(f"  Key pos words: {', '.join(list(analysis['key_words_pos'])[:5])}")
        if analysis['key_words_neg']:
            print(f"  Key neg words: {', '.join(list(analysis['key_words_neg'])[:5])}")

    print("\n" + "=" * 80)
    print("QUESTION ANALYSIS")
    print("=" * 80)

    for analysis in all_question_analyses:
        print(f"\n{analysis['trait'].upper()}:")
        print(f"  Number of questions: {analysis['num_questions']}")
        print(f"  Avg question length: {analysis['avg_question_length']:.1f} words")
        print(f"  Question types: {analysis['factual_questions']} factual, {analysis['ethical_questions']} ethical, {analysis['personal_questions']} personal")
        print(f"  Sample: {analysis['sample_questions'][0][:60]}...")

    # Check for potential confounds
    print("\n" + "=" * 80)
    print("POTENTIAL CONFOUNDS")
    print("=" * 80)

    # Check if instruction length correlates with trait type
    length_ratios = [a['length_ratio'] for a in all_instruction_analyses]
    if max(length_ratios) / min(length_ratios) > 2:
        print("\n⚠️  WARNING: Large variation in instruction length ratios")
        print(f"   Range: {min(length_ratios):.2f} - {max(length_ratios):.2f}")
        print("   This could mean some traits are detected via length not content")

    # Check if certain traits use very different question types
    for analysis in all_question_analyses:
        total_typed = analysis['factual_questions'] + analysis['ethical_questions'] + analysis['personal_questions']
        if total_typed > 0:
            ethical_ratio = analysis['ethical_questions'] / total_typed
            if ethical_ratio > 0.5:
                print(f"\n⚠️  {analysis['trait'].upper()}: Heavy ethical question bias ({ethical_ratio:.0%})")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n✓ GOOD DESIGN CHOICES:")
    print("  - Same questions used for both pos/neg (eliminates topic bias)")
    print("  - Multiple instruction variations (5 per trait)")
    print("  - Clear opposing instructions for each trait")

    print("\n⚠️  POTENTIAL IMPROVEMENTS:")
    # Check verbosity specifically
    verb_analysis = next(a for a in all_instruction_analyses if a['trait'] == 'verbosity')
    if verb_analysis['length_ratio'] > 1.5:
        print(f"  - Verbosity: Pos instructions are {verb_analysis['length_ratio']:.1f}x longer than neg")
        print("    Could the model detect verbosity from instruction length alone?")

    # Check for traits with very simple questions
    simple_q_traits = [a['trait'] for a in all_question_analyses if a['avg_question_length'] < 5]
    if simple_q_traits:
        print(f"  - Traits with very simple questions: {', '.join(simple_q_traits)}")
        print("    Consider using more complex questions to test deeper understanding")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()