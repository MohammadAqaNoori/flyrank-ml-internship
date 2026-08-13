# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mohammad Aqa Noori
- **Lane:** Refresh / Content Opportunity Scoring
- **Decision supported:** Prioritize pages for content-refresh review
- **Repository:** https://github.com/MohammadAqaNoori/flyrank-ml-internship
- **Capstone notebook:** `work/notebooks/capstone.ipynb`

---

## 0. Abstract

This project asks whether historical search-query performance signals can help prioritize content items for content-refresh review. The analysis uses the FlyRank internship warehouse, with the final modeling table based on `fact_content_query_90d` and an evaluation population of 4,085 content items. A Random Forest model was trained using previous-30-day impressions, clicks, CTR, visible-query count, and rare-query count, and was compared with the Week-4 rule-based baseline using Precision@50. On the evaluation sample, the Random Forest achieved a Precision@50 of 0.86 compared with 0.82 for the baseline, an absolute improvement of 0.04. The resulting ranking is intended as decision-support for human content review, not as an automatic instruction to rewrite content or as evidence that refreshing a page will improve search performance.

---

# 1. Problem framing

## Research question

**Can historical search-query performance signals be used to rank content items for content-refresh review more effectively than a simple Week-4 rule-based baseline?**

## Decision supported

The analysis supports the decision:

> **Which content items should be prioritized for content-refresh review?**

The unit of analysis is a **content item**.

The output is a **ranked review queue**. Each content item receives a model score that can be used to prioritize human investigation.

The intended human action is to review the highest-ranked candidates first, inspect their historical search signals, and then decide whether a content refresh or additional investigation is appropriate.

The model does not automatically decide that content should be rewritten, deleted, merged, or otherwise changed.

## Why modeling helps

A content-review process may involve multiple historical search signals at the same time. A ranking model can combine these signals into a repeatable ordering of candidates.

The goal is therefore not to replace editorial judgment, but to make the initial review queue more focused and repeatable.

A wrong ranking can have two main costs:

- review time can be spent on content that does not require attention;
- potentially useful review opportunities can be placed too low in the queue.

For this reason, the capstone focuses on **ranking quality at the top of the queue**, using Precision@50 as the primary metric.

---

# 2. Data safety

## Data source

The analysis uses the **FlyRank internship warehouse**.

The final modeling table is:

```text
fact_content_query_90d
