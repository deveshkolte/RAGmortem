"""Generate fresh hidden holdout dataset in BioTrial clinical trial domain for Day 7 Phase B evaluation."""

import json
from pathlib import Path

from examples.reference_rag.app import ReferenceRagApp
from ragmortem.faults.models import (
    BaselineExecution,
    FaultConfig,
    FaultyExecution,
    InjectedFaultCase,
)

HIDDEN_DOCS_DIR = Path("examples/hidden_holdout_rag/documents")
QUESTIONS_OUT = Path("evals/hidden_holdout_questions.jsonl")
FAULTS_OUT = Path("evals/hidden_holdout_faults.jsonl")

ANSWERABLE_SPECS = [
    {
        "id": "bq01",
        "question": "Within how many calendar days must fatal or life-threatening SUSAR adverse reactions be reported to the FDA?",
        "gold_chunk_id": "chunk_biotrial_ind_safety_susar",
        "gold_answer": "No later than 7 calendar days after initial receipt of the information.",
    },
    {
        "id": "bq02",
        "question": "How many days following the IND anniversary date must the annual progress report be submitted?",
        "gold_chunk_id": "chunk_biotrial_ind_annual_summary",
        "gold_answer": "Within 60 calendar days of the anniversary date.",
    },
    {
        "id": "bq03",
        "question": "What is the statutory deadline for the FDA to review a sponsor's response to lift a clinical hold?",
        "gold_chunk_id": "chunk_biotrial_ind_clinical_hold",
        "gold_answer": "Exactly 30 calendar days to lift or maintain the hold.",
    },
    {
        "id": "bq04",
        "question": "How often must the independent Data Safety Monitoring Board meet to review interim safety data?",
        "gold_chunk_id": "chunk_biotrial_ind_dsmb_charter",
        "gold_answer": "Every 6 months to evaluate unblinded interim safety telemetry.",
    },
    {
        "id": "bq05",
        "question": "What prior approvals must a physician secure before administering an emergency compassionate use IND drug?",
        "gold_chunk_id": "chunk_biotrial_ind_expanded_access",
        "gold_answer": "Written confirmation of IRB chair concurrence and FDA telephone authorization.",
    },
    {
        "id": "bq06",
        "question": "How often must the sponsor review and revise the formal Investigator's Brochure?",
        "gold_chunk_id": "chunk_biotrial_ind_investigator_brochure",
        "gold_answer": "At least once every 12 months (annually).",
    },
    {
        "id": "bq07",
        "question": "What maximum blood volume collection in a 30-day window triggers mandatory prior IRB review for biomarker sub-studies?",
        "gold_chunk_id": "chunk_biotrial_ind_substudy_amendments",
        "gold_answer": "Collecting additional biological specimens exceeding 50 milliliters of whole blood.",
    },
    {
        "id": "bq08",
        "question": "What temperature threshold and duration excursion requires immediate quarantine of sensitive biologic shipments?",
        "gold_chunk_id": "chunk_biotrial_ind_lot_traceability",
        "gold_answer": "Exceeding 8 degrees Celsius for longer than 120 minutes.",
    },
    {
        "id": "bq09",
        "question": "Within how many hours must a study site notify the pharmacovigilance safety officer of a serious adverse event?",
        "gold_chunk_id": "chunk_biotrial_irb_prompt_reporting",
        "gold_answer": "Within 24 hours of first awareness.",
    },
    {
        "id": "bq10",
        "question": "How many days in advance of protocol expiration must an IRB renewal continuing review application be submitted?",
        "gold_chunk_id": "chunk_biotrial_irb_continuing_review",
        "gold_answer": "At least 45 calendar days prior to protocol expiration.",
    },
    {
        "id": "bq11",
        "question": "Within how many business days must an unanticipated problem involving risks to subjects (UPIRSO) be reported to the IRB?",
        "gold_chunk_id": "chunk_biotrial_irb_unanticipated_problems",
        "gold_answer": "Within 10 business days of determination.",
    },
    {
        "id": "bq12",
        "question": "What is the minimum number of voting members required on an Institutional Review Board roster?",
        "gold_chunk_id": "chunk_biotrial_irb_composition_quorum",
        "gold_answer": "At least 5 voting members representing diverse backgrounds.",
    },
    {
        "id": "bq13",
        "question": "At what age must pediatric clinical trial participants provide formal assent?",
        "gold_chunk_id": "chunk_biotrial_irb_vulnerable_populations",
        "gold_answer": "Children age 7 and older alongside double parental permission.",
    },
    {
        "id": "bq14",
        "question": "Within how many calendar days must major protocol deviations compromising subject safety be reported to the IRB?",
        "gold_chunk_id": "chunk_biotrial_irb_protocol_deviation",
        "gold_answer": "Within 5 calendar days.",
    },
    {
        "id": "bq15",
        "question": "Within how many hours must an investigator document the medical rationale after breaking a treatment blind code?",
        "gold_chunk_id": "chunk_biotrial_irb_emergency_unblinding",
        "gold_answer": "Within 2 hours (and notify sponsor within 12 hours).",
    },
    {
        "id": "bq16",
        "question": "What financial interest dollar threshold triggers mandatory conflict of interest disclosure under FDA Form 1572?",
        "gold_chunk_id": "chunk_biotrial_irb_financial_disclosure",
        "gold_answer": "Equity holdings or consulting royalties exceeding $50,000.",
    },
    {
        "id": "bq17",
        "question": "What is the duration of the dose-limiting toxicity observation window in Phase I dose escalation?",
        "gold_chunk_id": "chunk_biotrial_phase1_dose_limiting_toxicity",
        "gold_answer": "Precisely 28 consecutive days (4 weeks) following Cycle 1 Day 1 dose.",
    },
    {
        "id": "bq18",
        "question": "In a classic 3+3 trial design, how many DLTs in a cohort of 3 subjects permit immediate dose escalation?",
        "gold_chunk_id": "chunk_biotrial_phase1_3plus3_escalation",
        "gold_answer": "Zero of 3 evaluable subjects experiencing a DLT.",
    },
    {
        "id": "bq19",
        "question": "In Simon's two-stage Phase II design, how many objective responses among 15 Stage 1 patients prevent early futility closure?",
        "gold_chunk_id": "chunk_biotrial_phase2_futility_stopping",
        "gold_answer": "3 or more confirmed objective responses.",
    },
    {
        "id": "bq20",
        "question": "How many progression-free survival events are required to power the registrational Phase III study?",
        "gold_chunk_id": "chunk_biotrial_phase3_sample_size_power",
        "gold_answer": "A cumulative total of 320 progression-free survival (PFS) events.",
    },
    {
        "id": "bq21",
        "question": "What overall percent agreement is required to validate a companion diagnostic assay against the reference laboratory?",
        "gold_chunk_id": "chunk_biotrial_biomarker_companion_diagnostic",
        "gold_answer": "An overall percent agreement (OPA) of at least 95.0%.",
    },
    {
        "id": "bq22",
        "question": "Within how many days of approval must a sponsor initiate a post-marketing confirmatory Phase IV study?",
        "gold_chunk_id": "chunk_biotrial_post_marketing_phase4_commitments",
        "gold_answer": "Within 180 calendar days of commercial market authorization.",
    },
    {
        "id": "bq23",
        "question": "Within how many calendar days must stored biological biospecimens be destroyed upon subject withdrawal request?",
        "gold_chunk_id": "chunk_biotrial_withdrawal_biological_samples",
        "gold_answer": "Within 60 calendar days.",
    },
    {
        "id": "bq24",
        "question": "How many telephone calls and letters must be documented before classifying a subject as lost to follow-up?",
        "gold_chunk_id": "chunk_biotrial_lost_to_followup_criteria",
        "gold_answer": "At least 3 telephone contact attempts and at least 1 certified letter.",
    },
    {
        "id": "bq25",
        "question": "For how many years following drug marketing approval must trial investigators retain study records?",
        "gold_chunk_id": "chunk_biotrial_record_retention_period",
        "gold_answer": "A minimum period of 2 years following marketing approval (NDA/BLA).",
    },
]

UNANSWERABLE_SPECS = [
    {"id": "bq26_unans", "question": "What is the standard dietary caloric intake required for rodent toxicology studies prior to IND filing?"},
    {"id": "bq27_unans", "question": "Which HPLC column resin is mandated for analytical peptide synthesis purification?"},
    {"id": "bq28_unans", "question": "What is the reimbursement tariff paid to hospital anesthesiologists for pediatric sedations under Medicare?"},
    {"id": "bq29_unans", "question": "What is the maximum permissible shelf-life for freeze-dried bacterial plasmid expression vectors?"},
    {"id": "bq30_unans", "question": "Which European CE mark classification applies to disposable surgical laparoscopy scalpels?"},
    {"id": "bq31_unans", "question": "What is the maximum allowed dosage of ibuprofen in veterinary equine medicine?"},
    {"id": "bq32_unans", "question": "How many hours of continuing education in radiology must ultrasound technicians complete annually?"},
    {"id": "bq33_unans", "question": "What is the minimum air exchange rate per hour for hospital surgical suite negative pressure rooms?"},
    {"id": "bq34_unans", "question": "Which antibody isotype is utilized in rapid diagnostic lateral flow COVID-19 antigen cartridges?"},
    {"id": "bq35_unans", "question": "What is the legal statute of limitations for medical malpractice claims filed in the state of Ohio?"},
    {"id": "bq36_unans", "question": "What is the maximum allowable residual moisture content in commercial freeze-dried vaccine vials?"},
    {"id": "bq37_unans", "question": "Which laser wavelength is approved by the FDA for refractive corneal PRK surgery?"},
    {"id": "bq38_unans", "question": "How many square feet of dedicated cleanroom space is required for ISO Class 5 sterile compounding pharmacies?"},
    {"id": "bq39_unans", "question": "What is the standard incubation temperature for anaerobic blood culture bottles in microbiology testing?"},
    {"id": "bq40_unans", "question": "What is the statutory expiration period for pharmaceutical patents granted under the Hatch-Waxman Act?"},
    {"id": "bq41_unans", "question": "Which radioactive isotope is utilized in PET radiotracer myocardial perfusion imaging scans?"},
    {"id": "bq42_unans", "question": "What is the maximum permissible concentration of heavy metal lead in commercial children vitamins?"},
    {"id": "bq43_unans", "question": "How often must emergency crash cart defibrillator batteries undergo automated load testing?"},
    {"id": "bq44_unans", "question": "What is the minimum clinical observation time required following subcutaneous allergy immunotherapy injections?"},
    {"id": "bq45_unans", "question": "Which statistical software package is required for filing biological variance submissions with Health Canada?"},
    {"id": "bq46_unans", "question": "What is the mandatory quarantine duration for laboratory beagle dogs imported for safety testing?"},
    {"id": "bq47_unans", "question": "What is the maximum permissible endotoxin unit limit per kilogram per hour for parenteral infusions?"},
    {"id": "bq48_unans", "question": "Which surgical suture thread material is indicated for intracardiac prosthetic valve annuloplasty?"},
    {"id": "bq49_unans", "question": "What is the minimum annual salary paid to clinical trial monitors holding ACRP certification in Texas?"},
    {"id": "bq50_unans", "question": "What is the maximum allowable sound pressure level in intensive care unit neonatal incubators?"},
]

ABSTAIN_TEXT = "I do not have sufficient information in the provided context to answer this question."

HALLUCINATED_BIOTRIAL_RESPONSES = [
    "Rodent preclinical toxicology models enforce a strict dietary caloric regimen of 2,400 kcal/kg with 18% protein.",
    "Peptide chromatography requires C18 reverse-phase silica resin with 300-angstrom pore diameter.",
    "Hospital pediatric anesthesia encounters are reimbursed under Medicare code 00810 at $145.00 per 15-minute unit.",
    "Lyophilized plasmid vectors retain bio-activity for 36 months when stored at -80 degrees Celsius under nitrogen.",
    "Disposable laparoscopic surgical trocars are classified as Class IIb active surgical medical devices.",
    "Veterinary equine protocols administer oral flunixin meglumine up to 1.1 mg/kg every 24 hours.",
    "Ultrasound technologist credentials mandate 30 continuing medical education (CME) credit hours triennially.",
    "Negative pressure airborne infection isolation rooms enforce a minimum ventilation rate of 12 air changes per hour.",
    "Lateral flow rapid immunoassays utilize murine monoclonal IgG1 capture antibodies conjugated to colloidal gold.",
    "Medical liability tort claims in Ohio must be initiated within 1 calendar year from the date of alleged malpractice.",
    "Parenteral freeze-dried biologic formulations enforce a residual moisture limit not exceeding 2.0% via Karl Fischer.",
    "Excimer laser refractive surgery systems operate at a deep ultraviolet wavelength of 193 nanometers.",
    "Compounding cleanroom buffer zones require a minimum floor area of 100 square feet with laminar airflow hoods.",
    "Clinical anaerobic blood cultures incubate continuously at 35 degrees Celsius (+/- 1 degree) for 5 days.",
    "Pharmaceutical primary patents receive a statutory monopoly term of 20 years from the original priority filing date.",
    "Myocardial perfusion positron emission tomography utilizes Rubidium-82 chloride with a 75-second half-life.",
    "Pediatric nutritional supplements enforce an elemental lead threshold not exceeding 0.5 micrograms per daily serving.",
    "Automated external defibrillators (AED) execute automated battery and capacitor self-tests every 24 hours.",
    "Allergen immunotherapy subcutaneous injections require on-site clinical observation for a mandatory 30 minutes.",
    "Canadian health authority submissions mandate SAS version 9.4 with CDISC SEND formatted dataset packages.",
    "Canine purpose-bred preclinical subjects undergo a mandatory 14-day acclimation and quarantine inspection period.",
    "Sterile water for injection complies with USP endotoxin limits of not more than 0.25 Endotoxin Units (EU) per mL.",
    "Prosthetic heart valve annuloplasty rings are secured using 2-0 braided polyester sutures with pledget supports.",
    "Certified Clinical Research Associates in Texas receive an average compensation package of $98,500 annually.",
    "Neonatal intensive care incubator micro-environments are sound-insulated to maintain interior noise below 45 dBA.",
]


def main():
    app = ReferenceRagApp(corpus_dir=HIDDEN_DOCS_DIR, mock_mode=True)
    all_chunks = app.chunks
    all_chunk_ids = [c.id for c in all_chunks]

    # 1. Questions records
    questions_records = []
    for spec in ANSWERABLE_SPECS:
        questions_records.append({
            "id": spec["id"],
            "question": spec["question"],
            "gold_chunk_id": spec["gold_chunk_id"],
            "gold_answer": spec["gold_answer"],
            "type": "answerable",
        })
    for spec in UNANSWERABLE_SPECS:
        questions_records.append({
            "id": spec["id"],
            "question": spec["question"],
            "gold_chunk_id": None,
            "gold_answer": ABSTAIN_TEXT,
            "type": "unanswerable",
        })
    QUESTIONS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(QUESTIONS_OUT, "w", encoding="utf-8") as f:
        for r in questions_records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(questions_records)} hidden questions to {QUESTIONS_OUT}")

    # 2. Generate 100 Fault Cases (20 per category)
    cases: list[InjectedFaultCase] = []

    # Category 1: retrieval_miss (20 cases)
    for i in range(20):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        distractor_ids = [distractors[(i * 3 + j) % len(distractors)] for j in range(3)]
        distractor_scores = [round(0.19 - j * 0.03, 4) for j in range(3)]
        wrong_answer = f"Clinical procedures follow general institutional protocols: {distractor_ids[0]}."

        fault_exec = FaultyExecution(
            retrieved_ids=distractor_ids,
            gold_rank=None,
            answer=wrong_answer,
            scores=distractor_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=distractor_ids,
        )

        case = InjectedFaultCase(
            case_id=f"hidden_retrieval_{i+1:02d}",
            question_id=q_id,
            fault_type="retrieval_miss",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="retrieval_miss",
            expected_failure_condition=f"gold_chunk_id '{gold_id}' absent from top-3",
            actual_failure_condition=f"retrieved_ids={distractor_ids}, gold_present=False",
            validated=True,
            configuration={"fault_type": "retrieval_miss", "top_k": 3},
            metadata={"domain": "biotrial"},
        )
        cases.append(case)

    # Category 2: ranking_miss (20 cases)
    for i in range(20):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        top3_dist = [distractors[(i * 2 + j) % len(distractors)] for j in range(3)]
        other_dist = [distractors[(i * 2 + 3 + j) % len(distractors)] for j in range(2)]

        # Candidate pool of 6 chunks: top 3 distractors, rank 4 is gold, then 2 others
        cand_ids = top3_dist + [gold_id] + other_dist
        cand_scores = [0.825, 0.815, 0.810, 0.804, 0.785, 0.770]
        top_k_ids = cand_ids[:3]
        top_k_scores = cand_scores[:3]

        wrong_ans = f"Based on review, trial administration proceeds according to: {top_k_ids[0]}."

        fault_exec = FaultyExecution(
            retrieved_ids=top_k_ids,
            gold_rank=None,
            answer=wrong_ans,
            scores=top_k_scores,
            candidate_retrieved_ids=cand_ids,
            candidate_gold_rank=4,
            prompt_context_chunk_ids=top_k_ids,
        )

        case = InjectedFaultCase(
            case_id=f"hidden_ranking_{i+1:02d}",
            question_id=q_id,
            fault_type="ranking_miss",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="ranking_miss",
            expected_failure_condition=f"gold_chunk_id '{gold_id}' demoted to rank 4 outside top-3",
            actual_failure_condition=f"candidate_gold_rank=4, top_3={top_k_ids}",
            validated=True,
            configuration={"fault_type": "ranking_miss", "top_k": 3, "candidate_k": 6},
            metadata={"domain": "biotrial", "cutoff_gap": 0.006},
        )
        cases.append(case)

    # Category 3: generation_ignored_context (20 cases)
    GEN_ERRORS = [
        ("numerical_confusion", "Fatal SUSAR reactions must be reported to the FDA within 30 calendar days."),
        ("numerical_confusion", "The annual IND progress report must be submitted within 180 calendar days."),
        ("numerical_confusion", "The FDA has a statutory deadline of 90 calendar days to review a clinical hold response."),
        ("numerical_confusion", "The independent Data Safety Monitoring Board must meet every 24 months."),
        ("numerical_confusion", "The Investigator Brochure requires mandatory formal review every 5 years."),
        ("temporal_confusion", "Whole blood collection for biomarker sub-studies is evaluated across a 12-month calendar window."),
        ("temporal_confusion", "Continuous cold chain excursions exceeding 8 degrees Celsius are permitted up to 72 hours."),
        ("temporal_confusion", "Serious Adverse Events must be reported to the sponsor within 14 calendar days."),
        ("temporal_confusion", "Continuing review renewal applications must be submitted 5 business days prior to expiration."),
        ("entity_confusion", "UPIRSO adverse events must be reported to the local hospital billing board."),
        ("entity_confusion", "IRB review committees require voting membership composed entirely of certified clinical oncologists."),
        ("entity_confusion", "Pediatric assent is required for infants age 6 months and older."),
        ("entity_confusion", "Major protocol deviations are reported to the sponsor sales department."),
        ("distractor_attention", "The medical monitor documentation window is capped at 48 hours for emergency treatments."),
        ("distractor_attention", "Financial conflict of interest disclosure thresholds are set at $5,000."),
        ("answer_omission", "Dose-limiting toxicity observation spans a defined clinical window."),
        ("answer_omission", "The 3+3 decision matrix governs cohort expansion across clinical cohorts."),
        ("false_refusal", ABSTAIN_TEXT),
        ("false_refusal", ABSTAIN_TEXT),
        ("false_refusal", ABSTAIN_TEXT),
    ]

    for i in range(20):
        spec = ANSWERABLE_SPECS[i]
        q_id = spec["id"]
        gold_id = spec["gold_chunk_id"]

        base_chunks, base_scores = app.retrieve(spec["question"], k=3)
        base_exec = BaselineExecution(
            retrieved_ids=[c.id for c in base_chunks],
            gold_rank=1 if base_chunks and base_chunks[0].id == gold_id else None,
            answer=f"Baseline: {spec['gold_answer']}",
            scores=base_scores,
        )

        distractors = [cid for cid in all_chunk_ids if cid != gold_id]
        d1, d2 = distractors[(i * 3) % len(distractors)], distractors[(i * 3 + 1) % len(distractors)]
        ret_ids = [gold_id, d1, d2]
        scores = [0.74, 0.52, 0.46]
        err_type, gen_ans = GEN_ERRORS[i]

        fault_exec = FaultyExecution(
            retrieved_ids=ret_ids,
            gold_rank=1,
            answer=gen_ans,
            scores=scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=ret_ids,
        )

        case = InjectedFaultCase(
            case_id=f"hidden_generation_{i+1:02d}",
            question_id=q_id,
            fault_type="generation_ignored_context",
            question=spec["question"],
            gold_chunk_id=gold_id,
            gold_answer=spec["gold_answer"],
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="generation_ignored_context",
            expected_failure_condition=f"gold chunk '{gold_id}' at rank 1 but generator failed ({err_type})",
            actual_failure_condition=f"gold_rank=1, answer='{gen_ans[:50]}...'",
            validated=True,
            configuration={"fault_type": "generation_ignored_context", "variant": err_type},
            metadata={"domain": "biotrial", "variant": err_type},
        )
        cases.append(case)

    # Category 4: should_abstain (20 cases)
    for i in range(20):
        spec = UNANSWERABLE_SPECS[i]
        q_id = spec["id"]
        hallucinated_ans = HALLUCINATED_BIOTRIAL_RESPONSES[i]

        distractors = [all_chunk_ids[(i * 3 + j) % len(all_chunk_ids)] for j in range(3)]
        distractor_scores = [round(0.22 - j * 0.03, 4) for j in range(3)]

        base_exec = BaselineExecution(
            retrieved_ids=distractors,
            gold_rank=None,
            answer=ABSTAIN_TEXT,
            scores=distractor_scores,
        )

        fault_exec = FaultyExecution(
            retrieved_ids=distractors,
            gold_rank=None,
            answer=hallucinated_ans,
            scores=distractor_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=distractors,
        )

        case = InjectedFaultCase(
            case_id=f"hidden_abstain_{i+1:02d}",
            question_id=q_id,
            fault_type="should_abstain",
            question=spec["question"],
            gold_chunk_id=None,
            gold_answer=ABSTAIN_TEXT,
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="should_abstain",
            expected_failure_condition="question unanswerable from corpus; system hallucinated substantive answer",
            actual_failure_condition=f"retrieved_scores={distractor_scores}, answer='{hallucinated_ans[:50]}...'",
            validated=True,
            configuration={"fault_type": "should_abstain"},
            metadata={"domain": "biotrial"},
        )
        cases.append(case)

    # Category 5: unknown (20 cases)
    UNKNOWN_CASES = [
        # Missing scores (5)
        ("missing_scores_01", UNANSWERABLE_SPECS[20]["question"], [all_chunk_ids[0], all_chunk_ids[1]], None, "Quarantine times follow veterinary guidelines."),
        ("missing_scores_02", ANSWERABLE_SPECS[20]["question"], [all_chunk_ids[2], all_chunk_ids[3]], None, "Analytical validation percent agreement is documented in the validation binder."),
        ("missing_scores_03", UNANSWERABLE_SPECS[21]["question"], [all_chunk_ids[4], all_chunk_ids[5]], None, "Endotoxin units are metered by pharmacopeia rules."),
        ("missing_scores_04", ANSWERABLE_SPECS[21]["question"], [all_chunk_ids[6], all_chunk_ids[7]], None, "Post-marketing trials initiate per commercial schedule."),
        ("missing_scores_05", UNANSWERABLE_SPECS[22]["question"], [all_chunk_ids[8], all_chunk_ids[9]], None, "Intracardiac annuloplasty utilizes standard monofilament."),
        # Borderline scores [0.47, 0.53] (5)
        ("borderline_01", ANSWERABLE_SPECS[22]["question"], [all_chunk_ids[10], all_chunk_ids[11]], [0.505, 0.490], "Biospecimen destruction occurs following consent withdrawal."),
        ("borderline_02", UNANSWERABLE_SPECS[23]["question"], [all_chunk_ids[12], all_chunk_ids[13]], [0.502, 0.485], "Clinical trial monitor salaries vary by geographical market."),
        ("borderline_03", ANSWERABLE_SPECS[23]["question"], [all_chunk_ids[14], all_chunk_ids[15]], [0.508, 0.482], "Lost to follow-up diligence requires phone and mail attempts."),
        ("borderline_04", UNANSWERABLE_SPECS[24]["question"], [all_chunk_ids[16], all_chunk_ids[17]], [0.498, 0.475], "Incubator acoustic decibel ceilings conform to environmental specs."),
        ("borderline_05", ANSWERABLE_SPECS[24]["question"], [all_chunk_ids[18], all_chunk_ids[19]], [0.512, 0.480], "Study documentation retention is maintained for several years."),
        # Conflicting signals / multiple plausible causes (5)
        ("conflicting_01", ANSWERABLE_SPECS[0]["question"], [all_chunk_ids[0], all_chunk_ids[1]], [0.520, 0.485], "Fatal SUSAR notifications require 10 days."),
        ("conflicting_02", ANSWERABLE_SPECS[1]["question"], [all_chunk_ids[2], all_chunk_ids[3]], [0.515, 0.480], "Annual IND submissions occur within 90 days."),
        ("conflicting_03", UNANSWERABLE_SPECS[0]["question"], [all_chunk_ids[4], all_chunk_ids[5]], [0.525, 0.475], "Rodent feeding follows animal research standards."),
        ("conflicting_04", ANSWERABLE_SPECS[2]["question"], [all_chunk_ids[6], all_chunk_ids[7]], [0.518, 0.470], "Clinical hold determinations require 45 days."),
        ("conflicting_05", UNANSWERABLE_SPECS[1]["question"], [all_chunk_ids[8], all_chunk_ids[9]], [0.522, 0.465], "HPLC purification uses porous particles."),
        # Empty context window (5)
        ("empty_context_01", ANSWERABLE_SPECS[3]["question"], [], [], "DSMB review occurs biannually."),
        ("empty_context_02", UNANSWERABLE_SPECS[2]["question"], [], [], "Anesthesia reimbursement is set by local fee schedule."),
        ("empty_context_03", ANSWERABLE_SPECS[4]["question"], [], [], "Expanded access requires IRB and FDA signoff."),
        ("empty_context_04", UNANSWERABLE_SPECS[3]["question"], [], [], "Plasmid storage adheres to vendor instructions."),
        ("empty_context_05", ANSWERABLE_SPECS[5]["question"], [], [], "Investigator brochures are updated every year."),
    ]

    for i, (scen, q_text, ret_ids, ret_scores, gen_ans) in enumerate(UNKNOWN_CASES):
        base_exec = BaselineExecution(
            retrieved_ids=ret_ids,
            gold_rank=None,
            answer="Baseline unavailable.",
            scores=ret_scores,
        )

        fault_exec = FaultyExecution(
            retrieved_ids=ret_ids,
            gold_rank=None,
            answer=gen_ans,
            scores=ret_scores,
            candidate_retrieved_ids=None,
            candidate_gold_rank=None,
            prompt_context_chunk_ids=ret_ids,
        )

        case = InjectedFaultCase(
            case_id=f"hidden_unknown_{i+1:02d}",
            question_id=f"bq_unknown_{i+1:02d}",
            fault_type="unknown",
            question=q_text,
            gold_chunk_id=None,
            gold_answer="Unknown reference answer",
            baseline=base_exec,
            fault=fault_exec,
            expected_cause="unknown",
            expected_failure_condition="telemetry genuinely incomplete or ambiguous; root cause cannot be distinguished",
            actual_failure_condition=f"scenario={scen}, scores={ret_scores}",
            validated=True,
            configuration={"fault_type": "unknown", "scenario": scen},
            metadata={"domain": "biotrial", "ambiguity_reason": scen},
        )
        cases.append(case)

    FAULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(FAULTS_OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c.to_dict()) + "\n")

    print(f"Total hidden holdout cases generated: {len(cases)}")
    from collections import Counter
    counts = Counter(c.fault_type for c in cases)
    for ft, count in counts.items():
        print(f"  {ft}: {count}")

if __name__ == "__main__":
    main()
