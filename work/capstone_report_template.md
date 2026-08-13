# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Mohammad Aqa Noori
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/MohammadAqaNoori/flyrank-ml-internship
- **Notebook:** `work/notebooks/capstone.ipynb`
- **Date:** 2026-08-13

---

## 0. Abstract

This capstone asks whether a simple machine-learning ranking model can prioritize content items for refresh review more effectively than a transparent rule-based baseline. The analysis uses a public-safe slice of the FlyRank internship warehouse, with the final modeling table taken from `fact_content_query_90d` and aggregated to the content level. A Random Forest uses five historical search-performance features and ranks content by the estimated probability of belonging to an observed impression-decline class. On the held-out test set, the Random Forest achieved 0.86 Precision@50 compared with 0.82 for the Week-4 baseline, an absolute improvement of 0.04. The resulting ranking is intended as decision-support for human content review, not as an automatic instruction to refresh content or a claim about future traffic outcomes.

---

## 1. Problem framing

### Research question

Can a simple machine-learning ranking model use currently available content and search-performance signals to prioritize pages for content-refresh review more effectively than a simple rule-based baseline?

### Decision supported

The analysis supports the decision of which content items should be reviewed first for a possible content refresh.

The output is a ranked decision-support queue. A higher-ranked item receives a higher model score and is therefore a stronger candidate for human review.

The model does **not** determine that a page definitely needs a refresh, nor does it establish that refreshing the page will improve search performance.

### Unit of analysis

The unit of analysis is a content item represented by a pseudonymous `content_hash_id`.

The final query-level data is aggregated to the content level before modeling.

### Output

The model produces a ranking score for each evaluated content item. Items with higher predicted probability for the observed decline class appear higher in the review queue.

### Human action

A content reviewer can use the ranking to:

1. Start with the highest-ranked content items.
2. Review their historical impressions, clicks, CTR, and query visibility.
3. Determine whether the observed decline is meaningful enough to investigate.
4. Inspect the content manually.
5. Decide whether a refresh or another action is appropriate.

### Cost of a wrong call

A high-ranked item may not actually require a refresh, creating unnecessary review work. A low-ranked item may contain a meaningful opportunity that the ranking did not surface. For that reason, the output is decision-support rather than an automatic action system.

### Why ML helps

The baseline provides a simple ranking rule based only on previous-30-day impressions. The Random Forest can combine several historical signals and model nonlinear relationships between them. The purpose of the comparison is therefore to test whether the additional modeling complexity improves the ranking of observed decline cases under the same evaluation conditions.

---

## 2. Data safety

### Data source

The capstone uses the FlyRank internship warehouse.

The final modeling table is:

`fact_content_query_90d`

The notebook also inspected a 100,000-row streaming slice of `fact_content_daily_performance` during data exploration. This exploratory slice was used to inspect available warehouse signals and was not the final modeling table. :contentReference[oaicite:2]{index=2}

### Final modeling data

The `fact_content_query_90d` table contains query-level search-performance information aggregated over 90-day windows.

For the working query-table slice:

- Rows loaded: 100,000
- Columns: 21
- Window start: 2026-04-02
- Window end: 2026-06-30
- Unique content items after aggregation: 5,648

The query-level records were aggregated to the content level using previous-30-day impressions, last-30-day impressions, previous-30-day clicks, last-30-day clicks, visible query count, rare query count, and content-level 90-day impressions. :contentReference[oaicite:3]{index=3}

### Evaluation population

To reduce noise from extremely low-volume content, only content items with at least 10 previous-30-day impressions were retained.

This produced:

- **4,085 evaluation items**
- **2,898 observed declines**
- **1,187 stable/increased items**
- **70.94% positive target rate**
- **29.06% non-positive target rate**

The 10-impression threshold is a noise-control rule rather than a claim that 10 impressions is a universal business threshold. :contentReference[oaicite:4]{index=4}

### Target definition

A content item is labeled positive when:

1. It has at least 10 impressions in the previous 30-day period.
2. Its impressions in the subsequent 30-day period are lower than its previous-30-day impressions.

The target therefore represents an **observed impression-decline signal**.

It does not establish why the decline happened and does not mean that the content itself caused the decline. :contentReference[oaicite:5]{index=5}

### Predictive features

The Random Forest uses five historical features:

- `impressions_prev30`
- `clicks_prev30`
- `ctr_prev30`
- `visible_query_count`
- `rare_query_count`

`ctr_prev30` is calculated from previous-30-day clicks and impressions only. All five features have zero missing values in the final evaluation population. :contentReference[oaicite:6]{index=6}

### Deliberate exclusions

The following fields are excluded from predictive inputs:

- `client_hash_id`
- `content_hash_id`
- `query_hash_id`
- `impressions_last30`
- `clicks_last30`
- `target`

The future-window outcome fields are used only to construct the evaluation target and are not supplied to the model. :contentReference[oaicite:7]{index=7}

### Leakage protection

The analysis applies the following safeguards:

- Future 30-day impression outcomes are not model features.
- The evaluation target is not a model feature.
- Pseudonymous client, content, and query identifiers are not predictive features.
- Previous-30-day CTR uses only previous-30-day clicks and impressions.
- The baseline and model are evaluated on the same held-out test population.
- Precision@50 is calculated using the same target for both approaches. :contentReference[oaicite:8]{index=8}

### Public safety

The public-facing analysis does not expose client names, URLs, private queries, credentials, or other client-identifying information.

---

## 3. Baseline

The Week-4 baseline ranks content items by their previous-30-day impressions.

This is intentionally simple and transparent. It represents a rule that could be implemented without a machine-learning model.

The baseline is a fair comparison because:

- It uses historical information available before the evaluated outcome.
- It produces the same type of ranking output as the model.
- It is evaluated on exactly the same held-out test set.
- It uses the same target.
- It uses the same Precision@50 metric.

### Baseline result

The baseline achieved:

- **Precision@50: 0.82**
- **Positive cases in top 50: 41**

The baseline therefore identified 41 of the 50 highest-ranked test items as belonging to the observed decline class. :contentReference[oaicite:9]{index=9}

---

## 4. Model / analysis

### Method

The selected method is a **Random Forest classifier**.

The model's predicted probability for the positive class is used as the ranking score. Items with higher predicted probability are ranked higher for review.

Random Forest was selected because it can represent nonlinear relationships among a small set of numerical search-performance features without requiring a complex modeling pipeline. :contentReference[oaicite:10]{index=10}

### Model configuration

The model used:

- `n_estimators = 200`
- `max_depth = 8`
- `min_samples_leaf = 10`
- `random_state = 42`
- `n_jobs = -1`
- `class_weight = "balanced"`

### Features

The exact predictive feature list is:

1. `impressions_prev30`
2. `clicks_prev30`
3. `ctr_prev30`
4. `visible_query_count`
5. `rare_query_count`

The model does not use future-window outcomes, identifiers, or the target as predictive inputs. :contentReference[oaicite:11]{index=11}

### Target

The target is an observed decline between the previous 30-day and subsequent 30-day impression windows, after applying the minimum threshold of 10 previous-30-day impressions.

This is an evaluation label rather than a claim about future refresh success.

---

## 5. Evaluation

### Split design

The 4,085-item evaluation population was divided using a:

- **75/25 train/test split**
- **Stratified by target**
- **random_state = 42**

This produced:

- Training rows: **3,063**
- Testing rows: **1,022**

Training target rate:

- **70.94%**

Testing target rate:

- **70.94%**

The stratification keeps the target proportion approximately consistent between training and testing. :contentReference[oaicite:12]{index=12}

### Important evaluation qualification

This is a controlled held-out comparison, but it is **not a time-series backtest**.

Therefore, the result should be interpreted as evidence from this evaluation sample rather than proof that the same performance will occur on future unseen time periods. :contentReference[oaicite:13]{index=13}

### Metric

The primary metric is:

**Precision@50**

Precision@50 measures the proportion of the 50 highest-ranked items that belong to the observed decline class.

The evaluation population has a positive-class base rate of **70.94%**.

This base rate is important because a high Precision@50 should not be interpreted without considering how common the positive class already is.

### Model vs baseline

| Method | Precision@50 | Positive cases in top 50 |
|---|---:|---:|
| Week-4 baseline | 0.82 | 41 |
| Random Forest | **0.86** | **43** |

The Random Forest improved Precision@50 by:

**0.04 absolute**, or **4 percentage points**.

Relative to the baseline Precision@50, this is approximately a **4.9% relative improvement**.

The Random Forest therefore placed two additional observed decline cases in the top 50 compared with the baseline under the same test-set conditions. :contentReference[oaicite:14]{index=14}

### Comparison with the base rate

The positive-class base rate is:

**70.94%**

Compared with this base rate:

- Week-4 baseline: **82%**, approximately 11.06 percentage points above the base rate.
- Random Forest: **86%**, approximately 15.06 percentage points above the base rate.

This indicates that both ranking approaches outperform simply selecting items without ranking information, while the Random Forest provides a further improvement over the Week-4 rule.

### Error analysis

The top-20 model ranking provides both successful and unsuccessful examples.

For example, the first ten model-ranked items in the recorded evaluation output are all target-positive cases. However, the ranking is not perfect: several lower-ranked items in the top 20 have target `0`, including items at positions 11 and 13.

This shows that the model can prioritize many observed decline cases successfully, but it also produces false positives.

The appropriate interpretation is therefore:

> The model improves the focus of the review queue; it does not perfectly identify every content item that will decline.

---

## 6. Interpretation

### Feature importance

The Random Forest feature importance output was:

| Feature | Importance |
|---|---:|
| `impressions_prev30` | 0.378722 |
| `rare_query_count` | 0.278914 |
| `visible_query_count` | 0.235016 |
| `ctr_prev30` | 0.077027 |
| `clicks_prev30` | 0.030321 |

The largest model importance in this evaluation sample was assigned to previous-30-day impressions.

Rare-query count and visible-query count were also substantial contributors to the model's ranking behavior.

CTR and previous-30-day clicks had smaller feature-importance values in this fitted model. :contentReference[oaicite:15]{index=15}

### What this means

The model appears to rely most heavily on the scale of previous search exposure and the breadth/structure of query visibility available in the modeled data.

However, feature importance describes model reliance in this particular fitted sample. It does **not** establish that any feature causes impression declines.

### Negative result / caution

The Random Forest's improvement over the baseline is meaningful for this evaluation, but it is modest:

- Baseline: 0.82 Precision@50
- Model: 0.86 Precision@50
- Improvement: 0.04

This means the model adds useful ranking signal, but the result does not justify claiming that a complex model is dramatically better than a simple rule.

The strongest practical finding is therefore that a small feature set plus a Random Forest produced a somewhat more focused review queue than the Week-4 baseline under the tested conditions.

---

## 7. Recommendation

The model output should be used as a **ranked content-review queue**.

### Recommended action playbook

#### Priority 1 — Review the highest-ranked items

Start with the highest model scores because they are the items the model assigns the greatest probability of belonging to the observed decline class.

#### Priority 2 — Verify the evidence manually

For each high-ranked item, inspect:

- Previous-30-day impressions
- Previous-30-day clicks
- Previous-30-day CTR
- Visible query count
- Rare-query count
- Relevant historical context available to the reviewer

#### Priority 3 — Decide whether the decline deserves action

A high model score should trigger investigation, not an automatic rewrite.

The reviewer should determine whether:

- The decline is substantial.
- The item has enough search visibility to justify review.
- The pattern appears meaningful.
- A content refresh is actually appropriate.

#### Priority 4 — Separate review from action

The ranking should not automatically trigger:

- rewriting,
- deletion,
- publishing,
- SEO changes,
- or other irreversible actions.

Human review remains necessary.

#### Priority 5 — Use lower-confidence items for monitoring

Items with lower model scores or unusual patterns can be monitored rather than immediately prioritized for refresh.

### Recommended workflow

1. Open the highest-ranked review candidates.
2. Inspect their historical search-performance signals.
3. Check whether the observed decline is meaningful.
4. Review the content manually.
5. Decide whether refresh, monitoring, or another action is appropriate.
6. Record the decision for future evaluation.

### Confidence

The strongest claim supported by this work is:

> **The Random Forest produced a better Precision@50 ranking than the Week-4 baseline on the same held-out evaluation set.**

The work does **not** support the stronger claims that:

- every high-ranked page needs a refresh,
- refreshing a page will recover impressions,
- the model identifies causal reasons for decline,
- or the measured improvement will necessarily reproduce on another dataset or future time period.

---

## 8. Reproducibility

### Repository

The project repository is:

https://github.com/MohammadAqaNoori/flyrank-ml-internship

The main capstone notebook is:

`work/notebooks/capstone.ipynb`

The report is:

`work/capstone_report.md`

### Notebook workflow

The capstone notebook performs the following sequence:

1. Loads a manageable streaming slice of `fact_content_daily_performance` for exploration.
2. Loads a 100,000-row streaming slice of `fact_content_query_90d`.
3. Aggregates query-level records to the content level.
4. Creates previous/last-30-day impression and click measures.
5. Restricts the evaluation population to items with at least 10 previous-30-day impressions.
6. Creates the observed impression-decline target.
7. Builds the five historical predictive features.
8. Performs the reproducible 75/25 stratified split with `random_state=42`.
9. Computes the Week-4 baseline.
10. Trains the Random Forest.
11. Generates model probabilities and the ranked queue.
12. Compares model and baseline using Precision@50.
13. Produces the top-20 review output.
14. Generates feature-importance and ranking artifacts.
15. Saves the paper artifacts.

### Reproducibility settings

The train/test split uses:

`random_state=42`

The Random Forest also uses:

`random_state=42`

The model configuration is recorded directly in the capstone notebook.

### Data access

The notebook uses the FlyRank internship warehouse through the Hugging Face `datasets` workflow with streaming enabled.

The notebook intentionally uses manageable slices rather than attempting to load the entire warehouse into memory.

A valid Hugging Face read-access configuration/token is required to reproduce the data-loading cells. Credentials must never be committed to the repository.

### Artifacts

The notebook generates:

- `model_vs_baseline.csv`
- `ranked_recommendations.csv`
- `feature_importance.csv`

These artifacts support the charts and tables used in the research paper. :contentReference[oaicite:16]{index=16}

### Environment note

The analysis uses Python notebook tooling including:

- pandas
- NumPy
- scikit-learn
- Matplotlib
- Hugging Face `datasets`

No credentials are stored in the repository.

### Reproducibility limitation

The evaluation is reproducible from the recorded notebook logic and random seeds, but it should not be described as a sealed blind evaluation or a time-series backtest. The evaluation is a 75/25 stratified random split over the defined evaluation population.

---

## 9. Acknowledgments & data credit

Built on the **FlyRank ML Internship dataset**.

Data source:

https://flyrank.ai

This project was completed as part of the FlyRank ML Internship and uses the internship warehouse for the capstone analysis.

---

# Capstone closing materials

## 5-minute demo outline

### 0:00–0:45 — Problem

Explain that the capstone addresses the problem of prioritizing content items for possible refresh review.

The goal is not to automatically rewrite content, but to make the human review queue more focused.

### 0:45–1:30 — Data

Show the `fact_content_query_90d` workflow and explain that query-level records are aggregated to content level.

Show the 4,085-item evaluation population and the 10-impression threshold.

### 1:30–2:15 — Method

Explain the five historical features and the Random Forest ranking approach.

Show that the future outcome is used only for evaluation and is not a predictive input.

### 2:15–3:15 — Baseline vs model

Show the comparison:

- Week-4 baseline: 0.82 Precision@50
- Random Forest: 0.86 Precision@50

Explain that the model puts 43 observed decline cases in the top 50 compared with 41 for the baseline.

### 3:15–4:15 — Interpretation

Show the feature-importance chart.

Explain that previous-30-day impressions had the largest model importance, followed by rare-query count and visible-query count.

Emphasize that feature importance is not causal evidence.

### 4:15–5:00 — Recommendations and limitations

Show the ranked review queue.

Explain the human-review workflow and finish with the main limitation:

The model identifies observed decline patterns in this evaluation sample; it does not prove that a content refresh will recover performance.

---

## Social-post cut

I built a machine-learning ranking model for the FlyRank ML Internship capstone to prioritize content items for refresh review.

The model uses five historical search-performance signals and was compared against a simple Week-4 baseline on the same held-out test set.

The Random Forest achieved **0.86 Precision@50**, compared with **0.82** for the baseline — a **4 percentage-point improvement**.

The important part is the framing: this is decision-support for human content review, not a claim that the model knows why a page declined or that a refresh will automatically recover traffic.

---

## Employer-facing summary

I built a small end-to-end machine-learning ranking system that turns historical search-performance signals into a prioritized content-review queue.

I compared a Random Forest against a transparent rule-based baseline using the same held-out test population and Precision@50, achieving 0.86 versus 0.82.

The project demonstrates my approach to practical ML: define a decision first, protect against leakage, establish a simple baseline, evaluate honestly, interpret the model cautiously, and turn the result into an actionable workflow.

---

# Claims checklist

Before submitting the capstone, all public claims should follow these rules:

- Use **observed** when describing the evaluation target.
- Use **measured** when describing model or baseline metrics.
- Use **directional** when describing model interpretation.
- Use **decision-support** when describing the ranked recommendations.
- Do not claim causality.
- Do not claim that a refresh will recover traffic.
- Do not claim to predict Google's algorithm.
- Do not expose client names.
- Do not expose private URLs.
- Do not expose private queries.
- Do not expose credentials.
- Do not expose raw private exports.
- Report the **70.94% base rate** alongside Precision@50.
- Keep the model result tied to the defined evaluation sample.
- Do not describe the 75/25 random split as a time-series backtest.
- Do not claim generalization to future periods without additional validation.

---

# Final self-check

- [x] Lane selected: Refresh / Content Opportunity Scoring
- [x] Decision defined: prioritize content for refresh review
- [x] Final modeling table identified: `fact_content_query_90d`
- [x] Exploratory data source documented
- [x] Evaluation population defined
- [x] Target definition documented
- [x] Predictive features documented
- [x] Future outcome fields excluded from predictive inputs
- [x] Pseudonymous identifiers excluded from predictive inputs
- [x] Baseline documented
- [x] Random Forest documented
- [x] Train/test split documented
- [x] Random seed documented
- [x] Leakage safeguards documented
- [x] Base rate reported
- [x] Model vs baseline reported on the same test set
- [x] Precision@50 reported
- [x] Error analysis included
- [x] Feature importance interpreted cautiously
- [x] Ranked recommendations documented
- [x] Artifacts documented
- [x] Reproducibility section included
- [x] Acknowledgments and FlyRank data credit included
- [x] Demo outline included
- [x] Social-post cut included
- [x] Employer-facing summary included
- [x] Claims use careful, non-causal language
- [x] No client-identifying information included
