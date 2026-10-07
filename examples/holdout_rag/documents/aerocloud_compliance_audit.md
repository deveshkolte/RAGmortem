# AeroCloud Platform: Regulatory Compliance, Security Auditing, and Data Governance

## [chunk_aerocloud_soc2_type2_reporting] SOC 2 Type II Certification and Audit Windows
AeroCloud undergoes annual SOC 2 Type II examination conducted by an independent AICPA-accredited auditing firm covering the Security, Availability, and Confidentiality Trust Services Criteria. The assessment testing period spans a rolling 12-month window from November 1 through October 31 of each calendar year. Formal attestation reports are finalized and made available to Enterprise customers under non-disclosure agreement (NDA) via the Trust Center by December 15 annually.

## [chunk_aerocloud_hipaa_baa_eligibility] HIPAA Compliance and Business Associate Agreements
Enterprise tier customers handling Protected Health Information (PHI) are eligible to execute a standard Business Associate Agreement (BAA) with AeroCloud. Under the BAA, PHI may only be processed within designated HIPAA-compliant cloud regions (us-east-1, us-west-2, and eu-west-1) with dedicated KMS encryption keys enabled. Standard tier and free promotional accounts are strictly prohibited from storing or transmitting PHI, and BAA execution is unavailable on non-Enterprise plans.

## [chunk_aerocloud_gdpr_data_subject_rights] GDPR Data Subject Rights and Erasure Timelines
In accordance with GDPR Article 17 (Right to Erasure), verified Data Subject Access Requests (DSAR) submitted through the Privacy API or admin console are processed across active production databases within 14 calendar days. Cryptographically sealed immutable backup archives retain deleted identifiers for a maximum of 30 additional days before automated cryptographic key shredding renders all residual records irrecoverable, ensuring full compliance within the statutory 30-day GDPR window.

## [chunk_aerocloud_audit_log_immutable_storage] Immutable Audit Trail Retention and WORM Storage
All control plane administrative operations, IAM credential mutations, and data access events are captured by the AeroCloud Audit Trail subsystem. Audit log records are streamed into Write-Once-Read-Many (WORM) compliant S3 Object Lock storage with compliance mode enabled. Audit logs cannot be modified, truncated, or deleted by any user or root principal for a mandatory retention period of 7 years (2,555 days).

## [chunk_aerocloud_vulnerability_disclosure_sla] Responsible Vulnerability Disclosure and Patch SLAs
AeroCloud maintains a public vulnerability coordination program with established remediation SLAs based on Common Vulnerability Scoring System (CVSS v3.1) metrics. Critical vulnerabilities (CVSS 9.0–10.0) require initial triage within 4 hours and deployment of a hotfix within 24 hours. High severity vulnerabilities (CVSS 7.0–8.9) require resolution within 7 calendar days, while Medium severity items (CVSS 4.0–6.9) are remediated within 30 calendar days.

## [chunk_aerocloud_pci_dss_tokenization] PCI-DSS Level 1 Scope and Tokenization Engine
AeroCloud is certified as a PCI-DSS Level 1 Service Provider. Primary Account Numbers (PAN) and sensitive authentication data are intercepted at the perimeter by an isolated tokenization vault and replaced with non-reversible surrogate tokens. AeroCloud microservices and persistent storage tiers never store, transmit, or process raw cardholder PAN data in plaintext, significantly reducing customer compliance scope under SAQ D.

## [chunk_aerocloud_disaster_recovery_rpo_rto] Business Continuity, RPO, and RTO Objectives
AeroCloud guarantees a Recovery Point Objective (RPO) of not more than 5 minutes for relational databases and 0 minutes (synchronous multi-AZ replication) for persistent block storage. The Recovery Time Objective (RTO) across multi-region failover automation is certified at under 30 minutes for core compute and data ingress pathways. Regional disaster recovery drill simulations are executed semi-annually under simulated network partition conditions.

## [chunk_aerocloud_third_party_vendor_review] Third-Party Vendor Security and Supply Chain Review
All third-party SaaS vendors and sub-processors with logical access to customer metadata undergo annual security evaluations based on the CAIQ (Consensus Assessments Initiative Questionnaire) standard. Vendor SOC 2 Type II or ISO 27001 certifications must remain active. AeroCloud publishes material changes to its sub-processor roster at least 30 calendar days prior to granting new vendor credentials, allowing customers to lodge formal objections.

## [chunk_aerocloud_fedramp_moderate_boundary] FedRAMP Moderate Authorization Boundary
Government and defense sector customers requiring FedRAMP Moderate authorization must deploy within the isolated AeroCloud GovZone sovereign cloud enclave. GovZone enforces FIPS 140-3 validated cryptographic modules, NIST SP 800-53 Rev. 5 baseline controls, and restricts infrastructure operations exclusively to screened US citizens located on domestic soil. Interconnection with commercial AeroCloud multi-tenant regions is blocked at the physical network gateway.
