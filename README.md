# AI-Era Data Challenge — Lending Club (Intern)

## Objective
You are advising a retail investor choosing among **newly listed** Lending Club loans. Build a small, reproducible pipeline that:
1) cleans and explores the data,  
2) trains a baseline model to predict default risk **at listing time**,  
3) converts probabilities into an investment choice under a budget, and  
4) **backtests** that policy on a held-out quarter.

Optimize for clarity over perfection. The scope is intentionally intern-level and time-boxed.

---

## Data
Quarterly CSVs are provided in `data/` (2016Q1–2017Q4). A data dictionary is in `docs/`.
Treat each quarter’s listings as a decision window.

**Listing-time only rule:** Use only information known when a loan is first listed. Do **not** use post-origination outcomes or fields that directly/indirectly reveal them (examples: `loan_status`, `last_pymnt_d`, `last_pymnt_amnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `out_prncp`, `next_pymnt_d`, any `*_rec_*`, any `*_pymnt*`, any `chargeoff*`, any `settlement*`).

> Tip: If unsure, ask yourself: “Could this value exist before the first payment was ever made?” If not, it’s disallowed.

---

## Tasks
1. **EDA & Cleaning**
   - Handle types, missing values, and obvious outliers.
   - Document any dropped columns and why.
   - Produce a quick data summary (rows, columns, target prevalence).

2. **Feature Set (Listing-time safe)**
   - Engineer features that could be known at listing (e.g., loan amount, interest rate, term, applicant employment info, FICO range).
   - Provide a short **feature provenance table** (a few representative features) explaining why each is valid at listing time.

3. **Baseline Model & Evaluation**
   - Train a simple classifier (e.g., logistic regression, tree/GBM) that outputs calibrated probabilities of default (PD).
   - Use a **time-ordered split**: train on earlier quarters, validate on the next quarter (e.g., train: 2016Q1–2016Q3, validate: 2016Q4).  
   - Report: ROC-AUC **and** calibration (reliability curve) **and** Brier score on the validation set.
   - Briefly interpret calibration (e.g., “over-confident in the 0.1–0.2 bin”).

4. **Decision Policy & Budget**
   - With a **$5,000 budget per quarter**, select loans based on your predicted risk (e.g., top-K with lowest PD or highest expected value).
   - Spell out your rule (threshold or ranking). Count selected loans per quarter.

5. **Backtest**
   - Apply your selection rule to a **held-out later quarter** (e.g., 2017Q1).  
   - Report at least:
     - **Selected default rate vs. overall default rate** in that quarter.
     - A simple **ROI proxy**. You may choose a reasonable, documented proxy. For example:

       ```text
       ROI_proxy = (collected_payments - principal) / principal
       where, for a toy assumption:
         if not default: collected_payments ≈ installment * term_months
         if default: collected_payments ≈ 0.30 * installment * term_months   # assume 30% paid before default
       ```

     - State your assumptions clearly; simpler is fine if well-explained.

6. **Explainability**
   - Show top features (coefficients or feature importance) and note one or two surprising relationships.

7. **(Optional, +5 pts) Tiny “AI-era” Extension**
   - Add **one** light text-derived feature (e.g., `emp_title` length, contains digits/keywords). Show if it helped.

---

## Deliverables (via GitHub PR)
- A single **notebook** (or `.py` script) that runs end-to-end locally.
- `SUMMARY.md` (≤1 page) with: approach, metrics, assumptions, decision rule, what you’d try next.
- `requirements.txt` with pinned versions sufficient to run your code.
- A short **AI-use disclosure** (if used): where you used AI assistance (e.g., boilerplate EDA) and how you validated the output.

**Timebox:** Aim for **4–6 hours** total.

---

## Guardrails & Checks
- **Listing-time only:** Do not use post-event/banned fields (see examples above).  
- **Temporal validation:** Ensure `max(issue_d)` in train < `min(issue_d)` in validation.  
- **Reproducibility:** Fix random seeds where applicable and include requirements.  
- **Calibration:** Include a reliability plot and Brier score on the hold-out/validation set.  
- **Decision policy:** Make your budget rule explicit and backtest it on a later quarter.

> Submissions that rely on random splits, leak obvious outcome fields, or skip the decision/backtest step will be marked down.

---

## Suggested Scoring (100 pts)
- **Data hygiene & EDA (20)** – sensible cleaning, types, missingness, clear notes  
- **Leakage avoidance (15)** – listing-time feature discipline; caught obvious traps  
- **Modeling & calibration (20)** – baseline model with PDs; calibration + interpretation  
- **Decision & backtest (20)** – coherent rule, budget applied, metrics reported  
- **Reasoning & communication (15)** – clear `SUMMARY.md`; trade-offs & next steps  
- **AI-use transparency (5)** – where/why AI was used and how validated  
- **Optional extension (5)** – one small text feature with measured effect

---

## Notes
- You may downsample/upsample or use class weights; explain your choice.
- Keep the solution compact and readable. We care about your reasoning as much as your metrics.
- If you propose a different, sensible ROI proxy or selection rule, that’s fine—just document it.


## Submission Information

### Create a fork of the repository
<img width="1699" height="393" alt="image" src="https://github.com/user-attachments/assets/f887255c-0155-4b0e-b90b-5d42b26b73b6" />

---

<img width="1792" height="870" alt="image" src="https://github.com/user-attachments/assets/1719eb7a-e2d2-4db9-8241-150a5f911f3d" />

---

This will create a forked copy of the repository under your github account. You can use this fork to create to complete your data challenge. When your submission is ready, you can open a pull request (PR) against this repository using the branch on your fork that contains your work. Your fork has knowledge of the main repository it was created from. Use the `Contribute` button to create the PR.

<img width="1745" height="777" alt="image" src="https://github.com/user-attachments/assets/43c55bc0-eccd-4108-aa12-53c8796efd05" />

---


