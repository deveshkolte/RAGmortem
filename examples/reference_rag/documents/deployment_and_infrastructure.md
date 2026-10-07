# Deployment & Infrastructure Operational Standards
<!-- Reference synthetic dataset for RAGmortem evaluation -->

## [chunk_deployment_freeze_window] Production Deployment Freeze Windows
To maintain system stability over weekends, all routine production deployments and database schema migrations are strictly prohibited after 2:00 PM local time on Fridays. Deployments during this window require explicit VP of Engineering override.

## [chunk_canary_duration] Canary Deployment Soak Duration
All production releases must undergo progressive canary rollouts. A canary release begins by directing 5% of production traffic to the new version and must soak for a minimum duration of 45 minutes before proceeding to 100% rollout.

## [chunk_rollback_threshold] Automated Canary Rollback Criteria
During the 45-minute canary soak phase, if the automated monitoring system detects an HTTP 5xx error rate exceeding 0.5% or a p99 latency increase greater than 20%, an automated rollback is immediately executed.

## [chunk_cicd_timeout] CI/CD Pipeline Execution Time Limit
To prevent runner starvation, all CI/CD pipeline jobs have an enforced hard execution timeout limit of 25 minutes. Any build job exceeding 25 minutes is automatically aborted and marked as failed.

## [chunk_database_backup_retention] Production Database Backup Retention
Automated full database snapshots are taken daily at 03:00 UTC. These snapshots are encrypted and retained for 35 days in geographically distributed cloud object storage.

## [chunk_rto_tier1] Tier-1 Recovery Time Objective (RTO)
All tier-1 critical microservices are required to satisfy a maximum Recovery Time Objective (RTO) of 15 minutes in the event of an infrastructure region failure.

## [chunk_rpo_tier1] Tier-1 Recovery Point Objective (RPO)
All tier-1 data stores and replication pipelines must satisfy a Recovery Point Objective (RPO) of 5 minutes, ensuring minimal data loss during failover scenarios.

## [chunk_dr_drill_frequency] Disaster Recovery Drill Cadence
The engineering infrastructure team must execute full disaster recovery and regional failover drills on a quarterly cadence (every three months) to validate recovery procedures.

## [chunk_mtls_enforcement] Inter-Service Mutual TLS Specification
All communication between backend services across the service mesh must enforce Mutual TLS (mTLS) with TLS version 1.3. Plaintext HTTP traffic between internal microservices is rejected.

## [chunk_api_rate_limit] Public REST API Rate Limiting Policy
Public-facing REST API gateways enforce a default rate limit of 120 requests per minute per IP address. Exceeding this threshold results in an HTTP 429 Too Many Requests response.
