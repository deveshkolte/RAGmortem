"""Generate cross-domain evaluation dataset (FinDebt: Syndicated Credit Agreement) for Day 8.

Dataset splits:
- development (20 cases: 5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- validation  (20 cases: 5 retrieval, 5 abstention, 5 terminology trap, 5 ambiguous)
- final_holdout (40 cases: 10 retrieval, 10 abstention, 10 terminology trap, 10 ambiguous)

Total: 80 cases in evals/abstention_holdout.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from examples.reference_rag.app import ReferenceRagApp

FINANCE_DOCS_DIR = Path("examples/finance_rag/documents")
HOLDOUT_OUT = Path("evals/abstention_holdout.jsonl")

# 20 Answerable questions (with verified gold chunks)
ANSWERABLE_SPECS = [
    {
        "qid": "fq01",
        "question": "What is the applicable interest rate margin over Term SOFR for Initial Term B borrowings?",
        "gold_chunk_id": "chunk_findebt_sofr_interest_margin",
        "gold_answer": "3.25% per annum for Term SOFR loans.",
    },
    {
        "qid": "fq02",
        "question": "What percentage of annual Excess Cash Flow must be applied toward mandatory prepayments?",
        "gold_chunk_id": "chunk_findebt_mandatory_excess_cash_flow",
        "gold_answer": "50% of annual Excess Cash Flow.",
    },
    {
        "qid": "fq03",
        "question": "What is the undrawn facility commitment fee rate for revolving credit commitments?",
        "gold_chunk_id": "chunk_findebt_commitment_fee_undrawn_facility",
        "gold_answer": "0.375% per annum on the unutilized portion.",
    },
    {
        "qid": "fq04",
        "question": "What is the maximum allowed Consolidated First Lien Net Leverage Ratio under the springing covenant?",
        "gold_chunk_id": "chunk_findebt_max_first_lien_leverage_ratio",
        "gold_answer": "4.25 to 1.00.",
    },
    {
        "qid": "fq05",
        "question": "What minimum Consolidated Interest Coverage Ratio must the borrower maintain?",
        "gold_chunk_id": "chunk_findebt_interest_coverage_ratio",
        "gold_answer": "Not less than 3.00 to 1.00.",
    },
    {
        "qid": "fq06",
        "question": "What is the monetary cross-default threshold on other indebtedness?",
        "gold_chunk_id": "chunk_findebt_cross_default_dollar_threshold",
        "gold_answer": "Principal amount exceeding $50,000,000.",
    },
    {
        "qid": "fq07",
        "question": "Within how many calendar days must the borrower deliver annual audited financial statements?",
        "gold_chunk_id": "chunk_findebt_annual_audited_financials_deadline",
        "gold_answer": "Within 90 calendar days after the end of each fiscal year.",
    },
    {
        "qid": "fq08",
        "question": "Within how many calendar days must quarterly unaudited financial reports be delivered?",
        "gold_chunk_id": "chunk_findebt_quarterly_unaudited_financials_deadline",
        "gold_answer": "Within 45 calendar days after the end of each fiscal quarter.",
    },
    {
        "qid": "fq09",
        "question": "What fair market value threshold triggers the requirement for a real property fee mortgage?",
        "gold_chunk_id": "chunk_findebt_real_property_mortgage_criteria",
        "gold_answer": "Fair market value exceeding $15,000,000.",
    },
    {
        "qid": "fq10",
        "question": "Within how many days after the Closing Date must Deposit Account Control Agreements be delivered?",
        "gold_chunk_id": "chunk_findebt_deposit_account_control_agreements",
        "gold_answer": "Within 60 calendar days of the Closing Date.",
    },
    {
        "qid": "fq11",
        "question": "What is the soft call prepayment premium percentage if a repricing event occurs within six months?",
        "gold_chunk_id": "chunk_findebt_voluntary_prepayment_call_protection",
        "gold_answer": "1.00% of the aggregate principal amount prepaid or repriced.",
    },
    {
        "qid": "fq12",
        "question": "What is the threshold of annual Net Cash Proceeds from asset sales before mandatory prepayment is required?",
        "gold_chunk_id": "chunk_findebt_asset_sale_sweep",
        "gold_answer": "Net Cash Proceeds exceeding $10,000,000 in any fiscal year.",
    },
    {
        "qid": "fq13",
        "question": "What additional default interest rate spread is added upon payment default?",
        "gold_chunk_id": "chunk_findebt_default_interest_spread",
        "gold_answer": "2.00% per annum above the rate otherwise applicable.",
    },
    {
        "qid": "fq14",
        "question": "On what anniversary of the Closing Date do the Initial Term B Loans mature?",
        "gold_chunk_id": "chunk_findebt_term_loan_maturity_date",
        "gold_answer": "The seventh (7th) anniversary of the Closing Date.",
    },
    {
        "qid": "fq15",
        "question": "What is the dollar cap on the annual general restricted payments basket?",
        "gold_chunk_id": "chunk_findebt_restricted_payments_basket",
        "gold_answer": "$25,000,000 per fiscal year.",
    },
    {
        "qid": "fq16",
        "question": "What is the maximum annual limit on Consolidated Capital Expenditures?",
        "gold_chunk_id": "chunk_findebt_annual_capital_expenditure_limit",
        "gold_answer": "$75,000,000 in any single fiscal year.",
    },
    {
        "qid": "fq17",
        "question": "How many business days of grace are permitted for payment of interest or fees?",
        "gold_chunk_id": "chunk_findebt_payment_default_grace_period",
        "gold_answer": "Five (5) business days after the scheduled due date.",
    },
    {
        "qid": "fq18",
        "question": "How many calendar days does the borrower have to cure an affirmative covenant default?",
        "gold_chunk_id": "chunk_findebt_covenant_breach_cure_period",
        "gold_answer": "Thirty (30) calendar days after written notice.",
    },
    {
        "qid": "fq19",
        "question": "What voting percentage constitutes Required Lenders for declaring acceleration of loans?",
        "gold_chunk_id": "chunk_findebt_lender_acceleration_remedies",
        "gold_answer": "Lenders holding more than 50.0% of the aggregate Loans and Commitments.",
    },
    {
        "qid": "fq20",
        "question": "What potential liability threshold triggers mandatory written notice of pending litigation?",
        "gold_chunk_id": "chunk_findebt_notice_of_litigation_threshold",
        "gold_answer": "Potential liability exceeding $15,000,000.",
    },
]

# 20 Genuine Abstention Questions (General finance/corporate, completely absent from credit agreement)
ABSTAIN_SPECS = [
    {"qid": "fq21_unans", "question": "What is the employer matching contribution percentage for the corporate 401(k) retirement plan?"},
    {"qid": "fq22_unans", "question": "What is the standard MACRS depreciation recovery period for corporate executive jet aircraft?"},
    {"qid": "fq23_unans", "question": "What is the maximum statutory corporate income tax rate applicable to corporate entities in Ireland?"},
    {"qid": "fq24_unans", "question": "What minimum FICO credit score is required for individuals applying for conforming residential mortgages?"},
    {"qid": "fq25_unans", "question": "Under European MiFID II directives, what is the maximum transaction reporting window for OTC derivatives?"},
    {"qid": "fq26_unans", "question": "What is the statutory vesting schedule for incentive stock options granted to corporate vice presidents?"},
    {"qid": "fq27_unans", "question": "How many weeks of paid maternity leave are mandated for salaried corporate headquarters personnel?"},
    {"qid": "fq28_unans", "question": "What is the per diem travel meal allowance for sales representatives traveling in continental Europe?"},
    {"qid": "fq29_unans", "question": "What is the minimum liquidity capital requirement for clearing member brokerage firms under Basel III?"},
    {"qid": "fq30_unans", "question": "What is the statutory expiration period for trademark registrations issued by the UK Intellectual Property Office?"},
    {"qid": "fq31_unans", "question": "What is the minimum statutory severance pay for employees terminated during corporate plant closures under the WARN Act?"},
    {"qid": "fq32_unans", "question": "What is the corporate discount rate applied for calculating lease liabilities under IFRS 16 rules?"},
    {"qid": "fq33_unans", "question": "What is the maximum annual contribution limit for Health Savings Accounts (HSA) under IRS guidelines?"},
    {"qid": "fq34_unans", "question": "Which actuarial mortality table is mandated for calculating defined benefit pension life annuities?"},
    {"qid": "fq35_unans", "question": "What is the statutory waiting period before exercising non-qualified employee stock appreciation rights?"},
    {"qid": "fq36_unans", "question": "What percentage of commercial real estate building value must be depreciated under Section 179 expensing?"},
    {"qid": "fq37_unans", "question": "What is the maximum allowable foreign tax credit carryforward period under US tax code Section 901?"},
    {"qid": "fq38_unans", "question": "How many days must an employee work before qualifying for group long-term disability insurance coverage?"},
    {"qid": "fq39_unans", "question": "What is the standard legal fee cap for patent prosecution filings before the European Patent Office?"},
    {"qid": "fq40_unans", "question": "What is the mandatory retention timeline for human resources I-9 employment eligibility forms?"},
]

# 20 Terminology-Overlap Trap Questions (Uses credit facility keywords but asks for absent criteria)
TRAP_SPECS = [
    {"qid": "fq41_trap", "question": "What is the interest rate margin discount if the borrower achieves ISO 14001 carbon neutrality metrics on Term SOFR borrowings?"},
    {"qid": "fq42_trap", "question": "What is the maximum allowed leverage ratio permitted when issuing cryptocurrency dividend distributions to shareholders?"},
    {"qid": "fq43_trap", "question": "Does the administrative agent charge a 5% prepayment penalty if the borrower refinances debt using municipal green bonds?"},
    {"qid": "fq44_trap", "question": "Under what conditions does an SEC Form 10-K delay trigger an automatic physical liquidation of manufacturing facilities?"},
    {"qid": "fq45_trap", "question": "What is the maximum balance permitted in cryptocurrency cold-storage wallets under the Deposit Account Control Agreement?"},
    {"qid": "fq46_trap", "question": "What is the mandatory excess cash flow sweep percentage if the company generates negative operating cash flows?"},
    {"qid": "fq47_trap", "question": "What is the revolving credit commitment fee rate if the company issues convertible subordinated preferred stock?"},
    {"qid": "fq48_trap", "question": "Does failure to submit the annual operating budget within 30 days trigger an automatic foreclosure on fee-owned real estate?"},
    {"qid": "fq49_trap", "question": "What is the required lender voting threshold to approve debt incurrence for acquiring sovereign foreign bonds?"},
    {"qid": "fq50_trap", "question": "What is the soft call protection premium if the Term Loan is prepaid using proceeds from an initial public offering in Tokyo?"},
    {"qid": "fq51_trap", "question": "Does an environmental cleanup citation exceeding $10,000,000 require an immediate 100% cash prepayment of all revolving credit loans?"},
    {"qid": "fq52_trap", "question": "What is the Consolidated Interest Coverage Ratio requirement if the borrower operates commercial nuclear reactors?"},
    {"qid": "fq53_trap", "question": "Can the borrower purchase equity interests from shareholders if the purchase is funded by issuing sovereign gold certificates?"},
    {"qid": "fq54_trap", "question": "Does a payment default on Term B Loans grant the Administrative Agent authority to sell the CEO's personal residential property?"},
    {"qid": "fq55_trap", "question": "What is the maximum fee-owned real property mortgage value if the property is located on a federally protected wetland?"},
    {"qid": "fq56_trap", "question": "Within how many calendar days must the CFO compliance certificate be delivered if the quarterly financial model contains AI hallucinations?"},
    {"qid": "fq57_trap", "question": "Does the cross-default threshold apply to default on employee corporate travel credit card accounts?"},
    {"qid": "fq58_trap", "question": "What is the required lender vote to reduce the default interest rate spread to zero percent?"},
    {"qid": "fq59_trap", "question": "How many business days does the borrower have to notify the agent if a director is cited for a minor traffic violation?"},
    {"qid": "fq60_trap", "question": "Does the asset disposition sweep apply if the borrower sells scrap paper and recycled office furniture?"},
]

# 20 Ambiguous / Borderline Questions
AMBIGUOUS_SPECS = [
    {"qid": "fq61_ambig", "question": "What is the grace period for administrative defaults?"},
    {"qid": "fq62_ambig", "question": "What is the required lender vote to amend loan terms?"},
    {"qid": "fq63_ambig", "question": "Can the borrower acquire foreign subsidiaries without prior lender consent?"},
    {"qid": "fq64_ambig", "question": "What constitutes a permitted intercompany merger among loan parties?"},
    {"qid": "fq65_ambig", "question": "When does the springing financial covenant become untestable?"},
    {"qid": "fq66_ambig", "question": "What is the procedure for releasing collateral upon an affiliate transfer?"},
    {"qid": "fq67_ambig", "question": "How are unutilized capital expenditure allowances allocated between fiscal years?"},
    {"qid": "fq68_ambig", "question": "What events permit lenders to terminate the revolving credit commitment early?"},
    {"qid": "fq69_ambig", "question": "What constitutes an arm's-length transaction with corporate affiliates?"},
    {"qid": "fq70_ambig", "question": "What are the legal enforceability opinion requirements for subsidiary guarantors?"},
    {"qid": "fq71_ambig", "question": "What happens if the Consolidated Total Net Leverage exceeds 3.50x during a dividend payment?"},
    {"qid": "fq72_ambig", "question": "Under what circumstances may the Administrative Agent waive a payment default grace period?"},
    {"qid": "fq73_ambig", "question": "What is the maximum allowed general debt basket if the leverage ratio is 4.00x?"},
    {"qid": "fq74_ambig", "question": "What is the timeline for delivering financial statements if the SEC grants an extension?"},
    {"qid": "fq75_ambig", "question": "Are intellectual property security interests perfected before recording at the USPTO?"},
    {"qid": "fq76_ambig", "question": "What constitutes beneficial ownership under the USA PATRIOT Act verification rules?"},
    {"qid": "fq77_ambig", "question": "What notice is required if a lender assigns its loan commitment to another bank?"},
    {"qid": "fq78_ambig", "question": "What is the definition of Material Subsidiary for cross-default purposes?"},
    {"qid": "fq79_ambig", "question": "Can the borrower execute voluntary prepayments while a technical default notice is pending?"},
    {"qid": "fq80_ambig", "question": "What constitutes a Repricing Event under the soft call protection clause?"},
]


def main():
    app = ReferenceRagApp(corpus_dir=FINANCE_DOCS_DIR, mock_mode=True)
    print(f"Loaded finance app with {len(app.chunks)} chunks.")

    all_cases: list[dict[str, Any]] = []

    # Helper to assign split
    def get_split(index: int) -> str:
        # 0..4 dev (5), 5..9 val (5), 10..19 holdout (10)
        if index < 5:
            return "development"
        elif index < 10:
            return "validation"
        else:
            return "final_holdout"

    # 1. 20 Retrieval Misses (Ground truth: retrieval_miss)
    for i, spec in enumerate(ANSWERABLE_SPECS):
        split = get_split(i)
        q = spec["question"]
        gold_id = spec["gold_chunk_id"]
        gold_ans = spec["gold_answer"]

        # Synthesize degraded retrieval missing the gold chunk
        all_ids = [c.id for c in app.chunks if c.id != gold_id]
        irrelevant_ids = all_ids[i * 2 : i * 2 + 3]

        all_cases.append({
            "case_id": f"findebt_retrieval_{i+1:02d}",
            "question_id": spec["qid"],
            "split": split,
            "fault_type": "retrieval_miss",
            "question": q,
            "gold_chunk_id": gold_id,
            "gold_answer": gold_ans,
            "fault": {
                "retrieved_ids": irrelevant_ids,
                "scores": [0.18, 0.15, 0.12],
                "candidate_retrieved_ids": None,
                "candidate_scores": None,
                "answer": f"Standard credit facility terms apply per {irrelevant_ids[0]}.",
                "prompt_context_chunk_ids": irrelevant_ids,
            },
            "expected_cause": "retrieval_miss",
            "metadata": {"domain": "findebt", "category": "retrieval_miss", "split": split},
        })

    # 2. 20 Genuine Abstention Cases (Ground truth: should_abstain)
    for i, spec in enumerate(ABSTAIN_SPECS):
        split = get_split(i)
        q = spec["question"]
        # Retriever retrieves whatever arbitrary chunks
        chunks, scores = app.retrieve(q, k=3)
        ret_ids = [c.id for c in chunks]

        all_cases.append({
            "case_id": f"findebt_abstain_{i+1:02d}",
            "question_id": spec["qid"],
            "split": split,
            "fault_type": "should_abstain",
            "question": q,
            "gold_chunk_id": None,
            "gold_answer": "I do not have sufficient information in the provided context to answer this question.",
            "fault": {
                "retrieved_ids": ret_ids,
                "scores": [round(s, 4) for s in scores],
                "candidate_retrieved_ids": None,
                "candidate_scores": None,
                "answer": "Under corporate guidelines, the requirement is 100% compliant with governing regulations.",
                "prompt_context_chunk_ids": ret_ids,
            },
            "expected_cause": "should_abstain",
            "metadata": {"domain": "findebt", "category": "genuine_abstain", "split": split},
        })

    # 3. 20 Terminology-Overlap Traps (Ground truth: should_abstain)
    for i, spec in enumerate(TRAP_SPECS):
        split = get_split(i)
        q = spec["question"]
        chunks, scores = app.retrieve(q, k=3)
        ret_ids = [c.id for c in chunks]

        all_cases.append({
            "case_id": f"findebt_trap_{i+1:02d}",
            "question_id": spec["qid"],
            "split": split,
            "fault_type": "should_abstain",
            "question": q,
            "gold_chunk_id": None,
            "gold_answer": "I do not have sufficient information in the provided context to answer this question.",
            "fault": {
                "retrieved_ids": ret_ids,
                "scores": [round(s, 4) for s in scores],
                "candidate_retrieved_ids": None,
                "candidate_scores": None,
                "answer": "Per credit facility guidelines, the required threshold is fully recognized.",
                "prompt_context_chunk_ids": ret_ids,
            },
            "expected_cause": "should_abstain",
            "metadata": {"domain": "findebt", "category": "terminology_trap", "split": split},
        })

    # 4. 20 Ambiguous Cases (Ground truth: unknown)
    for i, spec in enumerate(AMBIGUOUS_SPECS):
        split = get_split(i)
        q = spec["question"]
        chunks, scores = app.retrieve(q, k=3)
        ret_ids = [c.id for c in chunks]

        all_cases.append({
            "case_id": f"findebt_ambig_{i+1:02d}",
            "question_id": spec["qid"],
            "split": split,
            "fault_type": "unknown",
            "question": q,
            "gold_chunk_id": None,
            "gold_answer": None,
            "fault": {
                "retrieved_ids": ret_ids,
                "scores": [round(s, 4) for s in scores],
                "candidate_retrieved_ids": None,
                "candidate_scores": None,
                "answer": "The loan documents indicate various applicable procedures.",
                "prompt_context_chunk_ids": ret_ids,
            },
            "expected_cause": "unknown",
            "metadata": {"domain": "findebt", "category": "ambiguous", "split": split},
        })

    HOLDOUT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HOLDOUT_OUT, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c) + "\n")

    print(f"Wrote {len(all_cases)} cases to {HOLDOUT_OUT}")
    splits_summary = {}
    for c in all_cases:
        s = c["split"]
        splits_summary[s] = splits_summary.get(s, 0) + 1
    print("Splits summary:", splits_summary)


if __name__ == "__main__":
    main()
