# AeroCloud Platform: API Quotas, Concurrency, and Routing Policies

## [chunk_aerocloud_api_rate_limits] API Rate Limits and Burst Capacity
Standard tier API credentials allow a baseline throughput of 1,200 requests per minute (20 requests per second) per organization. Enterprise tier credentials scale to 12,000 requests per minute (200 requests per second). All API endpoints implement a token bucket algorithm allowing a maximum burst capacity of 150% of the baseline limit for sustained periods not exceeding 15 consecutive seconds. Requests exceeding burst thresholds receive an HTTP 429 Too Many Requests response with a mandatory `Retry-After` header specifying backoff duration in integer seconds.

## [chunk_aerocloud_idempotency_keys] Idempotency Key Semantics and TTL
Mutation operations via POST and PATCH accept an optional `Idempotency-Key` UUID v4 header. AeroCloud caches the initial response for exactly 86,400 seconds (24 hours). Subsequent requests submitted with an identical idempotency key return the cached HTTP status code, headers, and payload without triggering downstream service re-execution. If a mutation request is received while a prior request bearing the same idempotency key is actively processing, the gateway responds with HTTP 409 Conflict.

## [chunk_aerocloud_concurrency_quotas] WebSocket and Long-Polling Concurrency
Real-time telemetry connections via WebSocket or gRPC streaming are capped at 500 concurrent active channels per workspace on Standard plans and 10,000 concurrent channels on Enterprise plans. Connection attempts exceeding this quota are rejected with close frame code 4008 (Policy Violation). Heartbeat ping frames must be acknowledged within 30 seconds, or the edge proxy terminates the socket connection.

## [chunk_aerocloud_webhook_retry_backoff] Webhook Delivery and Exponential Backoff
AeroCloud dispatches asynchronous event notifications to customer-configured HTTPS webhook endpoints with an HMAC-SHA256 signature in the `X-Aero-Signature` header. If the destination endpoint returns a non-2xx HTTP code or times out after 5,000 milliseconds, the dispatch engine executes up to 7 automated retries using exponential backoff with jitter: 30s, 2m, 10m, 1h, 4h, 12h, and 24h. After 7 failed attempts, the webhook event is transferred to the Dead Letter Queue (DLQ).

## [chunk_aerocloud_api_deprecation_sla] API Versioning and Deprecation SLA
AeroCloud guarantees backward compatibility for all GA (General Availability) REST API versions for a minimum of 365 calendar days following an official deprecation notice. Deprecation warnings are announced via customer account notifications and transmitted in the `Sunset` HTTP response header according to RFC 8594. Beta and Preview API endpoints are not subject to the 365-day SLA and may be decommissioned with 30 calendar days notice.

## [chunk_aerocloud_pagination_limits] Query Pagination and Result Windows
List endpoints enforce cursor-based pagination utilizing opaque `starting_after` and `ending_before` cursors. The default page size across all REST collection endpoints is 50 items, and the maximum permissible page size is 250 items. Offset-based page numbering is strictly forbidden across production endpoints to prevent database index exhaustion on deep result scans.

## [chunk_aerocloud_service_mesh_timeouts] Internal Gateway Timeouts and Deadlines
Inbound HTTP client requests through the AeroCloud API gateway enforce a strict hard execution deadline of 29,000 milliseconds (29 seconds). If upstream microservices within the internal service mesh fail to produce an HTTP response before this deadline, the edge load balancer terminates the client connection and outputs HTTP 504 Gateway Timeout with an RFC 7807 problem details JSON payload.

## [chunk_aerocloud_bulk_export_quotas] Asynchronous Bulk Data Export Limits
Organizations may initiate up to 5 concurrent asynchronous bulk data export jobs per billing cycle. Export jobs generate compressed Parquet or JSONL archives written to customer-designated cloud storage buckets. Export job output files remain available for download for 72 hours before automatic lifecycle garbage collection permanently purges the temporary download URL.
