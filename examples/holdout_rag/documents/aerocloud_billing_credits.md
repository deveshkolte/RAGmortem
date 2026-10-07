# AeroCloud Platform: Billing, Invoicing, and Credit Entitlements

## [chunk_aerocloud_billing_cycle_settlement] Monthly Billing Cycles and Invoice Settlement
AeroCloud computes usage consumption across UTC calendar months, closing billing cycles at 23:59:59 UTC on the final day of each calendar month. Invoices are generated within 72 hours of cycle closure and charged automatically to the primary payment method on file. Customers on Net-30 payment terms must settle outstanding invoice balances within 30 calendar days from the invoice issuance timestamp to avoid automatic account degradation.

## [chunk_aerocloud_free_tier_credits] Promotional Credit Grants and Expiration Rules
New customer organizations receive a one-time promotional grant of $300 in AeroCloud trial credits valid for exactly 90 calendar days from tenant provisioning. Promotional credits automatically apply against compute runtime and storage usage, but cannot be deducted against marketplace third-party integrations, software licensing fees, or reserved instance commitments. Expired trial credits are non-refundable and forfeit immediately upon expiration.

## [chunk_aerocloud_compute_overage_rates] On-Demand Compute Overage Multipliers
Compute workloads exceeding contracted vCPU baseline allocations transition into on-demand overage billing without throttling active jobs. On-demand overage is metered at per-second granularity with a minimum billing increment of 60 seconds. On-demand compute rates carry a 20% premium surcharge over the baseline contracted unit rate.

## [chunk_aerocloud_dispute_resolution_window] Invoice Dispute and Billing Audit Timelines
Customers may formally dispute invoiced line items within 60 calendar days of invoice generation by submitting an audit inquiry through the billing support console. Undisputed invoice charges remain due according to standard terms. If an inquiry confirms metering overcharges, AeroCloud issues a credit memo applied against the immediate subsequent billing cycle. Cash refunds are only issued if the customer account has been permanently terminated.

## [chunk_aerocloud_reserved_capacity_cancellation] Reserved Capacity Commitments and Early Termination
Organizations purchasing 1-year or 3-year Reserved Cloud Instances receive up to 45% discount compared to on-demand rates. If a customer terminates a reserved commitment prior to the scheduled anniversary date, an early termination fee equal to 35% of the remaining contractual commitment balance is charged immediately to the account.

## [chunk_aerocloud_currency_conversion] Multi-Currency Billing and FX Conversion
All AeroCloud infrastructure consumption is denominated and ledger-settled in United States Dollars (USD). Invoices issued in supported local currencies (EUR, GBP, JPY, AUD) calculate conversion rates based on the Bloomberg BFIX 16:00 London fixing rate published on the final business day of the billing cycle, without adding retail currency exchange markups.

## [chunk_aerocloud_tax_withholding] International Tax Withholding and VAT Registration
Customers operating in the European Union or United Kingdom must supply a valid VAT or GST registration number during onboarding. In the absence of a verified tax identifier, AeroCloud automatically applies local destination country VAT rates to all invoiced digital services according to standard cross-border digital service tax regulations.

## [chunk_aerocloud_payment_failure_grace_period] Payment Default and Account Suspension Workflow
If an automated credit card charge fails, AeroCloud initiates a 14-calendar-day grace period. Automated retry attempts occur on day 3, day 7, and day 12. If payment is not secured by 23:59 UTC on day 14, compute instances are transitioned into stopped status. After 30 days of persistent non-payment, cloud storage volumes and snapshots are permanently deprovisioned.
