# Incident Management & Operational Response Policy
<!-- Reference synthetic dataset for RAGmortem evaluation -->

## [chunk_incident_severity_p1] P1 Critical Outage Definition and SLA
A Priority 1 (P1) incident is declared when a catastrophic outage affects more than 25% of active users or critical transaction processing is completely halted. The mandatory engineering response SLA for P1 incidents is 15 minutes from alert firing to incident triage.

## [chunk_incident_severity_p2] P2 Major Impairment Definition and SLA
A Priority 2 (P2) incident is declared when a core service feature is severely degraded but workarounds exist, affecting between 5% and 25% of users. The mandatory engineering response SLA for P2 incidents is 30 minutes.

## [chunk_incident_severity_p3] P3 Minor Impairment Definition and SLA
A Priority 3 (P3) incident is designated for non-critical bugs, localized errors, or minor performance degradations affecting less than 5% of users. The engineering team response SLA for P3 incidents is 2 hours during business hours.

## [chunk_incident_severity_p4] P4 Low Severity Maintenance and SLA
A Priority 4 (P4) incident encompasses cosmetic defects, documentation errata, or minor internal tool hiccups that do not disrupt end-user operations. P4 issues have an SLA requiring acknowledgement and assignment by the next business day.

## [chunk_incident_commander_role] Incident Commander Authority
During any active P1 or P2 incident, the designated Incident Commander (IC) possesses full unilateral authority over engineering operations, including the power to declare immediate production code freezes and command emergency service rollbacks.

## [chunk_postmortem_timeline] Blameless Postmortem Publication Schedule
Following the mitigation and closure of any P1 or P2 incident, the incident commander must lead a blameless postmortem analysis. The final postmortem report must be published to the engineering wiki within 48 hours of incident resolution.

## [chunk_communication_cadence] Executive Incident Communication Cadence
During an ongoing P1 incident, the communications lead must distribute internal executive status updates and customer incident page notifications at a minimum cadence of every 30 minutes until mitigation.

## [chunk_oncall_handoff] Primary On-Call Rotation Handoff Schedule
The engineering primary on-call rotation follows a weekly schedule. The formal handoff meeting and pager transfer between outgoing and incoming on-call engineers takes place every Tuesday at 10:00 AM UTC.

## [chunk_pagerduty_escalation] Automated On-Call Escalation Window
When an automated high-priority alert fires, the primary on-call engineer has 5 minutes to acknowledge the alert in PagerDuty. If the alert remains unacknowledged after 5 minutes, it automatically escalates to the secondary on-call engineer.
