# BioTrial Protocol: Good Clinical Practice (GCP), Electronic Data Capture, and Record Retention

## [chunk_biotrial_21cfr11_audit_trails] 21 CFR Part 11 Electronic Records and Computerized System Validation
All Electronic Data Capture (EDC) systems and clinical database management tools must fully comply with FDA 21 CFR Part 11 standards for electronic records and electronic signatures. The EDC platform enforces immutable, automated, cryptographically timestamped audit trails capturing the exact date, UTC time, user identity, prior data value, and newly submitted value for every individual case report form (eCRF) data field mutation.

## [chunk_biotrial_source_data_verification_sdv] Risk-Based Monitoring (RBM) and Source Data Verification (SDV)
Under ICH E6(R2) Good Clinical Practice guidelines, clinical monitors utilize a Targeted Risk-Based Monitoring plan where 100% Source Data Verification (SDV) is mandated for critical primary efficacy endpoints, eligibility confirmation checklists, and reported Serious Adverse Events (SAEs). Non-critical secondary laboratory panels and routine demographic records undergo sampled SDV across a minimum random 20% subject cohort.

## [chunk_biotrial_query_resolution_timelines] Clinical Data Management Query Resolution SLAs
When centralized data management or medical review teams issue automated or manual discrepancies (queries) regarding ambiguous eCRF entries, the site study coordinator or principal investigator must formally respond to or resolve the data query within 5 business days. Queries involving safety telemetry or concomitant medications receive priority resolution within 48 hours.

## [chunk_biotrial_tmf_inspection_readiness] Trial Master File (TMF) Archival and Inspection Readiness
The sponsor and clinical study sites must maintain an electronic Trial Master File (eTMF) structured in accordance with the DIA TMF Reference Model v3.0. Essential regulatory documents, investigator CVs, IRB approvals, and monitoring visit reports (MVRs) must be uploaded to the eTMF within 15 calendar days of final execution. Unannounced FDA bioresearch monitoring (BIMO) inspections require full TMF auditor access within 2 hours of inspector arrival.

## [chunk_biotrial_record_retention_period] Mandatory Clinical Trial Record Retention Lifecycles
Under 21 CFR 312.62(c), participating clinical investigators and sponsor organizations must retain all essential clinical study documents, case histories, and signed consent forms for a minimum period of 2 years following the date a marketing application (NDA/BLA) is approved for the drug indication. If no application is filed, records must be retained for 2 years after formal IND discontinuation and FDA notification.

## [chunk_biotrial_database_lock_criteria] Clinical Database Soft Lock and Final Hard Lock Criteria
Formal database hard lock terminating all data entry privileges occurs only after 100% of eCRF pages are entered, all data queries are resolved, all external safety and pharmacokinetic laboratory datasets are reconciled, and principal investigators have electronically signed all completed case books. Following hard lock, database unfreezing requires formal written authorization from both the Head of Biostatistics and the VP of Clinical Operations.

## [chunk_biotrial_blinded_sample_reconciliation] Central Laboratory Sample Reconciliation and Chain of Custody
All pharmacokinetic (PK) blood tubes and biomarker serum samples shipped to centralized analytical bio-laboratories must undergo physical barcode scanning and manifests reconciliation within 24 hours of site receipt. Discrepant tube labeling, broken dry ice seals, or missing specimens must be documented on a formal Sample Deviation Notice transmitted to the sponsor within 1 business day.

## [chunk_biotrial_investigator_site_file_closure] Close-Out Monitoring Visits and Site File Archiving
Upon completion of the final subject visit and resolution of all open data discrepancies, the Clinical Research Associate (CRA) conducts an on-site Close-Out Monitoring Visit (COV). The CRA verifies drug accountability reconciliation, confirms destruction or return of unused investigational product, and oversees final sealing of the paper Investigator Site File (ISF) into tamper-evident archival storage boxes.
