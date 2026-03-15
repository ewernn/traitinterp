---
title: "OOD: Formality Generalization"
preview: "Formality vectors generalize across languages (98.6%) and topics (96.4%)."
---

# OOD: Formality Generalization

Testing whether formality trait vectors generalize out-of-distribution across languages and topics.

## Setup

- **Trait:** Formality (formal vs casual language style)
- **Model:** Gemma-2-2B (base for extraction, IT for steering)
- **Cross-language:** English, Spanish, French, Chinese
- **Cross-topic:** Business, Academic, Social, Technical
- **Methods:** Classification cross-validation, steering cross-validation

### Example Scenarios

**Cross-language** (same formality concept, different languages):

:::dataset /datasets/traits/formality_variations/english/positive.txt "English formal":::

:::dataset /datasets/traits/formality_variations/spanish/positive.txt "Spanish formal":::

**Cross-topic** (same formality concept, different domains):

:::dataset /datasets/traits/formality_variations/business/positive.txt "Business formal":::

:::dataset /datasets/traits/formality_variations/social/positive.txt "Social formal":::

---

## Cross-Language Results

### Classification (Layer 12)

| Train \ Test | English | Spanish | French | Chinese |
|--------------|---------|---------|--------|---------|
| **English**  | 100.0%* | 100.0%  | 100.0% | 95.7%   |
| **Spanish**  | 100.0%  | 100.0%* | 100.0% | 91.3%   |
| **French**   | 100.0%  | 100.0%  | 100.0%*| 100.0%  |
| **Chinese**  | 95.7%   | 100.0%  | 100.0% | 95.7%*  |

- **In-domain:** 98.9%
- **Cross-domain:** 98.6% (+/- 2.7%)

### Steering (English -> Others, Layer 15, coef=100)

| Target  | Baseline | Steered | Delta  | Coherence |
|---------|----------|---------|--------|-----------|
| Spanish | 42.3     | 87.5    | +45.2  | 88%       |
| French  | 54.3     | 87.4    | +33.1  | 83%       |
| Chinese | 41.1     | 85.4    | +44.2  | 89%       |
| **Avg** |          |         | **+40.8** | **87%** |

### Key Findings

1. **Near-perfect classification transfer:** 98.6% cross-language accuracy shows formality is essentially language-independent in representation space.

2. **Symmetric transfer:** All languages transfer well to each other, with Chinese slightly weaker (95.7% in/out).

3. **Strong steering with high coherence:** English vector achieves +40.8 avg delta on other languages with 87% coherence (no degradation).

---

## Cross-Topic Results

### Classification (Layer 12)

| Train \ Test | Business | Academic | Social | Technical |
|--------------|----------|----------|--------|-----------|
| **Business** | 100.0%*  | 95.8%    | 100.0% | 89.5%     |
| **Academic** | 92.0%    | 100.0%*  | 100.0% | 84.2%     |
| **Social**   | 100.0%   | 100.0%   | 100.0%*| 94.7%     |
| **Technical**| 100.0%   | 100.0%   | 100.0% | 94.7%*    |

- **In-domain:** 98.7%
- **Cross-domain:** 96.4% (+/- 5.1%)

### Steering (Source -> Others, Layer 15, coef=100)

| Source -> | Business | Academic | Social | Technical | Avg |
|-----------|----------|----------|--------|-----------|-----|
| **Business** | -- | +35.4 | +10.8 | +31.7 | +26.0 |
| **Academic** | +25.0 | -- | +12.4 | +33.4 | +23.6 |
| **Social** | +25.5 | +32.3 | -- | +32.5 | +30.1 |

### Key Findings

1. **Strong classification transfer:** 96.4% cross-topic accuracy, with technical as the hardest target (84-95%).

2. **Good steering transfer:** All topic vectors achieve +24-30 avg delta on other topics with 87-89% coherence.

3. **Social has high baseline:** Social formality baseline is already ~76%, reducing headroom for delta.

---

## Classification vs Steering Transfer

| Dimension | Classification | Steering |
|-----------|---------------|----------|
| Cross-language | 98.6% | +40.8 delta, 87% coh |
| Cross-topic | 96.4% | +26.6 delta, 88% coh |

**Key insight:** Both classification and steering transfer cleanly across domains. The improved scoring prompts yield higher numbers than previous runs, with steering coherence remaining high across both language and topic transfer.

---

## Scripts

- `analysis/ood/cross_variant_evaluation.py` - Classification cross-validation
- `analysis/ood/cross_domain_steering.py` - Steering cross-validation

## Data

- `datasets/traits/formality_variations/{english,spanish,french,chinese}/` - Cross-language
- `datasets/traits/formality_variations/{business,academic,social,technical}/` - Cross-topic
- `datasets/traits/formality_variations/general/` - Combined (all topics)
