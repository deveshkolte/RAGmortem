# Security, Identity & Compliance Guidelines
<!-- Reference synthetic dataset for RAGmortem evaluation -->

## [chunk_vuln_critical_sla] Critical Vulnerability Patching SLA
Security vulnerabilities identified with a CVSS score of 9.0 or higher (designated as Critical) must be patched, remediated, or mitigated with a compensating control within 24 hours of notification.

## [chunk_vuln_high_sla] High Severity Vulnerability Remediation Window
Vulnerabilities evaluated with a CVSS score between 7.0 and 8.9 (designated as High) must be patched and deployed to production within 7 calendar days of verification.

## [chunk_secrets_rotation] Production Credential Rotation Frequency
All production database passwords, service account tokens, and cloud provider API keys must be automatically rotated every 90 days. Static hardcoded secrets in source repositories are strictly banned.

## [chunk_mfa_hardware_keys] Privileged SSH Hardware Key Requirement
Access to production infrastructure, bastion hosts, and Kubernetes control planes strictly requires multi-factor authentication via FIDO2-compliant hardware security keys (e.g. YubiKey). SMS and app-based TOTP are not permitted for production infrastructure access.

## [chunk_audit_log_retention] Security Audit Log Storage Duration
All cloud audit trails, authentication logs, and infrastructure modification events must be streamed to immutable storage and retained for 365 days to meet compliance standards.

## [chunk_session_timeout] Admin Console Inactivity Session Expiry
Administrative web consoles and cloud management dashboards enforce an idle session timeout of 15 minutes of inactivity, after which re-authentication is mandatory.

## [chunk_cab_approval] Change Advisory Board (CAB) Review Triggers
Any production modification affecting border gateway routing, edge firewall policies, or public DNS zones mandates prior review and formal approval from the Change Advisory Board (CAB).

## [chunk_data_encryption_at_rest] Cryptographic Standards for Data at Rest
All customer data stored in persistent disks, object stores, and database tables must be encrypted at rest using AES-256 with keys managed by the organization's Key Management Service (KMS).

## [chunk_pii_masking] Customer PII Anonymization in Non-Production
Customer personally identifiable information (PII) must never be loaded into staging or development environments. All database dumps replicated to non-production environments must undergo pseudonymization using SHA-256 with a unique salt.

## [chunk_external_vendor_review] Third-Party Vendor Risk Assessment Cadence
Any external software, SaaS platform, or third-party dependency with access to customer data must undergo a comprehensive vendor security risk assessment every 12 months.

## [chunk_least_privilege_review] IAM Access Permission Recertification
Cloud Identity and Access Management (IAM) role assignments and elevated privileges must be audited and recertified by team leads every 6 months to enforce least privilege principles.
