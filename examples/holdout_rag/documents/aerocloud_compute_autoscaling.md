# AeroCloud Platform: Compute Nodes, Autoscaling, and Resilience

## [chunk_aerocloud_autoscaling_cooldown] Autoscaling Cooldown Windows and Metric Polling
Compute autoscaling groups evaluate target metrics (CPU utilization, queue depth, HTTP latency) every 15 seconds. After an automated horizontal scaling event (adding or terminating nodes), the autoscaling engine enforces a mandatory cooldown period of 300 seconds (5 minutes) before executing subsequent scaling actions, preventing thrashing and oscillation.

## [chunk_aerocloud_warm_pool_provisioning] Pre-Warmed Compute Instance Pools
For latency-critical workloads requiring sub-second capacity expansion, organizations can configure Pre-Warmed Pools. Instances in the warm pool maintain initialized operating system kernels and cached container images in a stopped or paused state, reducing instance launch latency from 180 seconds (cold start) to under 30 seconds.

## [chunk_aerocloud_spot_instance_eviction] Spot Instance Interruption Notice Windows
AeroCloud Spot Instances provide up to 70% cost savings over on-demand compute instances in exchange for dynamic preemption when capacity is needed. When the orchestrator reclaims a spot instance, a 120-second (2-minute) preemption warning is broadcast via the internal instance metadata service at `http://169.254.169.254/latest/meta-data/spot/termination-time`.

## [chunk_aerocloud_node_drain_grace_period] Kubernetes Node Drain and Pod Termination Deadlines
When an instance is selected for automated retirement or maintenance termination, the cluster node drain controller issues a SIGTERM signal to all running application containers. Pods are allocated a maximum grace period of 90 seconds to finish in-flight requests and persist local state before the kubelet dispatches an unconditional SIGKILL signal.

## [chunk_aerocloud_multi_zone_high_availability] Multi-Zone Cluster High Availability SLAs
Production Kubernetes clusters configured across a minimum of three distinct Availability Zones within a region carry a 99.99% monthly control plane availability SLA. Single-zone clusters are classified as development-tier and carry a lower 99.5% availability target with zero financial credit eligibility for control plane outages.

## [chunk_aerocloud_hybrid_interconnect_latency] Dedicated Direct Interconnect Latency Targets
AeroCloud Cloud Interconnect provides private physical 10 Gbps and 100 Gbps fiber links connecting customer on-premises data centers directly to the AeroCloud software-defined backbone. Interconnect links guarantee round-trip network transit latency of less than 5 milliseconds between on-premises colocation facilities and adjacent cloud regions.

## [chunk_aerocloud_container_registry_rate_limits] Internal Container Registry Pull Quotas
The AeroCloud Container Registry (ACR) allows standard accounts to perform up to 5,000 container image layer pulls per hour. Image vulnerability scanning executes automatically on every image push, blocking deployments of container images containing known CVE vulnerabilities with a CVSS score of 9.0 or higher unless explicitly exempted by a `SecAdmin`.

## [chunk_aerocloud_service_disaster_recovery_rto] Disaster Recovery RTO and RPO Commitments
For Tier-1 mission-critical enterprise workloads with multi-region failover enabled, AeroCloud contractually commits to a Recovery Time Objective (RTO) of 1 hour and a Recovery Point Objective (RPO) of 5 minutes. Failover drills must be scheduled with technical account managers at least 14 days in advance.
