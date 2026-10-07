# AeroCloud Platform: Storage Tiers, Retention, and Lifecycle Policies

## [chunk_aerocloud_storage_tier_transitions] Object Storage Tier Classification and Auto-Tiering
AeroCloud Object Storage provides three access tiers: Standard (hot access, sub-10ms retrieval latency), Infrequent Access (IA, minimum 30-day retention duration, retrieval fees apply), and Archive Glacier (cold storage, minimum 90-day retention duration). Automated lifecycle rules transition objects from Standard to IA after 30 consecutive days of zero read operations, and to Archive Glacier after 90 days.

## [chunk_aerocloud_glacier_retrieval_windows] Glacier Archive Restoration Latencies
Restoring archived objects from the Archive Glacier tier supports three retrieval priorities: Expedited (restoration completed within 15 minutes, maximum object size 250 MB), Standard (restoration completed within 3 to 5 hours), and Bulk (restoration completed within 12 hours for batch datasets up to 10 TB). Restored temporary read copies remain active for a configurable window between 1 and 30 calendar days.

## [chunk_aerocloud_worm_object_locking] WORM Immutability and Compliance Retention Locks
Buckets configured with Object Lock enforce Write-Once-Read-Many (WORM) storage compliant with SEC Rule 17a-4. Under Compliance Mode, retained objects cannot be overwritten, modified, or deleted by any user, including root tenant administrators or AeroCloud support personnel, until the retention timestamp elapses. Under Governance Mode, privileged `SecAdmin` users may bypass retention with multi-factor authorization.

## [chunk_aerocloud_cross_region_replication] Cross-Region Bucket Replication (CRR) Lag SLA
Cross-Region Replication (CRR) asynchronously synchronizes object mutations, versions, and deletion markers between geographically distinct cloud regions. AeroCloud guarantees that 99.9% of object updates are replicated to destination regions within 900 seconds (15 minutes). Replication streams enforce TLS 1.3 encryption in transit with end-to-end SHA-256 checksum verification.

## [chunk_aerocloud_bucket_versioning_limits] Bucket Versioning and Soft Deletion Lifecycles
Enabling object versioning preserves historical iterations of overwritten or deleted items. Soft-deleted items receive an asynchronous deletion marker and remain retrievable for 30 calendar days before permanent cryptographic shredding. A single object key may store up to 1,000 distinct versions before the storage engine rejects subsequent overwrite attempts.

## [chunk_aerocloud_multipart_upload_cleanup] Incomplete Multipart Upload Expiration
When uploading large objects (>100 MB) utilizing the multipart upload protocol, uncommitted parts consume metered storage. If a multipart upload session is not finalized via a complete-multipart API call within 7 calendar days from initiation, the storage lifecycle engine automatically terminates the upload session and garbage collects all orphaned chunk parts.

## [chunk_aerocloud_block_volume_snapshots] Persistent Block Storage Snapshots and Quotas
Persistent block storage volumes attached to compute instances support crash-consistent automated point-in-time snapshots. Organizations may retain up to 250 automated snapshots per block volume. Snapshot differential data is compressed and replicated to multi-zone durable object storage with eleven nines (99.999999999%) of annual durability.

## [chunk_aerocloud_dedicated_throughput_limits] Dedicated Storage IOPS and Bandwidth Caps
High-performance SSD block volumes (io2 tier) deliver up to 64,000 Provisioned IOPS with a maximum throughput limit of 1,000 MB/sec per instance volume attachment. Burst credits accumulate during idle periods and allow sustained peak bursts for up to 30 continuous minutes per 24-hour cycle before dropping to baseline provisioned performance.
