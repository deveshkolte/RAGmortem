# AeroCloud Platform: IAM, Authentication, and Key Management

## [chunk_aerocloud_api_token_lifetimes] Service Account API Token Lifetimes
Service account programmatic bearer tokens generated via the AeroCloud IAM console have a configurable maximum time-to-live (TTL) of 90 calendar days. Static, non-expiring service tokens are prohibited by security policy. Ephemeral OAuth 2.0 access tokens exchanged via client credentials grants enforce a fixed validity window of 3,600 seconds (1 hour) and must be refreshed using rotating refresh tokens.

## [chunk_aerocloud_saml_sso_federation] Enterprise SAML 2.0 and OIDC Identity Federation
Enterprise subscriptions support single sign-on federation through SAML 2.0 and OpenID Connect (OIDC) identity providers, including Okta, Azure Active Directory, and Google Workspace. Just-In-Time (JIT) user provisioning is enabled by default. Session durations for federated identity assertions cannot exceed 12 hours before requiring re-authentication at the external identity provider.

## [chunk_aerocloud_session_revocation_sla] Emergency Session Revocation and Token Blacklisting
When an administrator revokes a user session or rotates an identity credential in the IAM console, revocation assertions propagate across all globally distributed edge authentication proxies within 120 seconds. Any active WebSocket streams or cached authorization assertions associated with the revoked session are immediately severed.

## [chunk_aerocloud_rbac_roles_hierarchy] Role-Based Access Control (RBAC) Built-in Roles
AeroCloud organizes access permissions into four built-in hierarchical roles: `OrgOwner` (unrestricted tenant authority and billing), `SecAdmin` (security policy, IAM, and audit log access), `DevOperator` (infrastructure provisioning, deployments, and cluster management), and `Auditor` (read-only visibility across logs and configuration). Custom roles can be composed with granular API action verbs.

## [chunk_aerocloud_ip_allowlisting] CIDR IP Allowlisting for Inbound Administration
Administrative console access and sensitive management API endpoints can be restricted to specific IPv4 and IPv6 CIDR blocks. Organizations may configure up to 50 distinct CIDR allowlist rules per tenant. Modifications to IP allowlists require multi-party approval from at least two distinct `SecAdmin` or `OrgOwner` accounts.

## [chunk_aerocloud_kms_envelope_encryption] KMS Envelope Encryption and Customer-Managed Keys (CMEK)
All customer data stored in AeroCloud volumes and object storage is encrypted at rest using AES-256-GCM envelope encryption. Organizations on Enterprise plans may integrate Customer-Managed Encryption Keys (CMEK) hosted in external KMS systems (AWS KMS, Azure Key Vault, or HashiCorp Vault). External master encryption keys must support automatic annual cryptographic rotation.

## [chunk_aerocloud_break_glass_emergency_access] Emergency Break-Glass Credentials
In the event of an identity provider outage or administrative lockout, organizations may activate the break-glass recovery account created during initial tenant onboarding. Break-glass credentials require a 40-character recovery master secret split across three cryptographic custodian shards using Shamir's Secret Sharing, requiring at least two shards to reassemble.

## [chunk_aerocloud_mtls_inter_service_encryption] Mutual TLS (mTLS) for Cluster Mesh Communication
All network traffic traversing the internal cluster network and cross-node communication links enforces mutual TLS (mTLS) utilizing TLS 1.3 with ephemeral Elliptic Curve Diffie-Hellman key exchange (ECDHE). Ephemeral node x509 identity certificates are automatically issued and rotated by the internal cluster PKI every 24 hours.

## [chunk_aerocloud_mfa_hardware_security_keys] Mandatory Multi-Factor Authentication (MFA) Enforcement
Organizations enforcing organizational security compliance must require all non-service interactive human accounts to enroll FIDO2/WebAuthn hardware security keys (e.g. YubiKey) or TOTP authenticator applications. SMS-based two-factor authentication is classified as deprecated and disallowed for accounts possessing `OrgOwner` or `SecAdmin` privileges.
