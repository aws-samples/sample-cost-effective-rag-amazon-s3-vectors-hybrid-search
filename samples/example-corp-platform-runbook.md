# Example-Corp Payments Platform - Operations Runbook

Internal reference for the Example-Corp Payments Platform team. Last revised: August 2026.
Owner: Platform Engineering (channel: #example-corp-platform-ops).

## 1. Platform overview

Example-Corp is the company's internal payments processing platform. It handles card
authorizations, refunds, and settlement batching for all consumer-facing
products. Example-Corp processes approximately 4.2 million transactions per day with
peaks of 180 transactions per second during regional sales events.

The platform consists of three planes:

- The **authorization plane** validates and routes card authorizations in real
  time through the acquiring partners (AnyCompany Acquiring and Example Organization).
- The **settlement plane** aggregates captured transactions into settlement
  batches, generated daily at 02:00 UTC.
- The **ledger plane** records every money movement in a double-entry ledger
  and is the system of record for finance reconciliation.

## 2. Service level objectives

| Metric | Objective | Measured over |
| --- | --- | --- |
| Authorization availability | 99.95% | Calendar month |
| Authorization latency (p99) | 350 milliseconds | Rolling 5 minutes |
| Settlement batch completion | Before 04:30 UTC | Daily |
| Refund processing time | Under 2 hours | Per request |
| Ledger reconciliation drift | Zero unexplained entries | Daily close |

If authorization availability drops below 99.9% mid-month, the on-call
engineering manager must open an incident review regardless of customer
impact, per the reliability charter.

## 3. Error code reference

- **ORN-201**: Authorization approved with elevated risk score. No action
  needed; logged for the fraud team's weekly review.
- **ORN-408**: Acquirer timeout. Example-Corp retries once with a 400 millisecond
  backoff. Sustained ORN-408 rates above 0.5% indicate an acquirer-side
  incident - check the AnyCompany Acquiring status page before escalating internally.
- **ORN-429**: Rate limiting engaged. The authorization plane sheds load when
  a single merchant exceeds 25 requests per second. Contact the merchant
  integrations team before raising limits.
- **ORN-500**: Internal processing failure. Automatically pages the on-call
  engineer when the 5-minute count exceeds 50 events.
- **ORN-601**: Settlement batch validation failure. The batch is quarantined;
  finance must approve a manual release. Never re-run a quarantined batch
  without finance sign-off.

## 4. Incident response

Severity levels:

- **SEV-1**: Authorization plane down or error rate above 5%. Page the
  platform on-call and the duty manager immediately. Public status page must
  be updated within 15 minutes.
- **SEV-2**: Degraded latency (p99 above 800 milliseconds for 10+ minutes) or
  settlement delay past 04:30 UTC. Page platform on-call.
- **SEV-3**: Single-merchant impact or elevated ORN-201 rates. Ticket only,
  business hours follow-up.

The on-call rotation is weekly, Monday 09:00 UTC handover, staffed from the
Dublin and Seattle offices. Escalation order: platform on-call, then duty
manager, then head of payments engineering.

During any SEV-1, freeze all deployments platform-wide. The deployment freeze
also applies every year from December 15 through January 5 (peak season), and
during the last business day of each fiscal quarter.

## 5. Deployment procedures

Example-Corp deploys through the deployment pipeline. Standard flow: merge to
main, automated canary to 5% of authorization traffic for 30 minutes, then
progressive rollout in 25% increments every 15 minutes. A canary is
automatically rolled back if the ORN-500 rate rises more than 0.2 percentage
points above the trailing baseline.

Manual rollbacks run through the deployment pipeline (`rollback --service <name> --to-release <id>`) and
complete in under 4 minutes. Rollbacks of the ledger plane additionally
require a ledger integrity check (`example-corpctl ledger verify --since 24h`)
before traffic is restored.

Database schema migrations must be backward compatible for at least one
release cycle. Destructive migrations require a two-week deprecation notice
posted in #example-corp-platform-announce.

## 6. Data retention and compliance

Authorization records are retained for 13 months in the hot store and 7 years
in cold archive, per PCI DSS and local financial regulations. Card numbers are
never stored - Example-Corp keeps only network tokens issued by the acquirers.
Ledger entries are immutable; corrections are posted as compensating entries,
never edits.

Access to production payment data requires the `example-corp-prod-reader` role,
granted only through a manager-approved request that expires after 90 days.
Quarterly access reviews are mandatory and tracked by the security team.

## 7. Frequently asked questions

**How do I request a new merchant rate limit?** Open a ticket with merchant
integrations including the merchant ID, current limit, requested limit, and
traffic justification. Standard turnaround is three business days.

**Why did my settlement batch not arrive?** Check for ORN-601 quarantine
first; then confirm the batch window (02:00 UTC generation, delivery to
banking partners by 04:30 UTC). Batches delayed past 04:30 UTC are a SEV-2.

**Who approves emergency changes during the December freeze?** Only the head
of payments engineering or the VP of infrastructure can approve a freeze
exception, and the change must have a tested rollback plan.

**Can Example-Corp process non-card payments?** Bank transfer support (SEPA and ACH)
is on the roadmap for Q2 2027 behind the `example-corp-alt-rails` feature flag; it
is not production-ready today.
