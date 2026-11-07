# Project B — Developer Operations Assistant  (Week 12)

Vault note: `Deepak_Library/AI Engineer/06-Projects/Project B - DevOps Assistant`

Investigates incidents using **read-only** tools: searches logs, retrieves runbooks,
summarises evidence with timestamps and links, drafts an incident report.
It never changes anything — every consequential action stops for human approval.

This project plays to your actual background: Elasticsearch, Kafka, Redis, production
support, release ownership. Build the thing only you can build.

## Non-negotiables
- bounded execution (hard step budget)
- explicit, serialisable state
- approval gates **in code**, not in the prompt
- full audit log
- evidence links with timestamps on every claim
- prompt-injection resistant (a hostile log line cannot fire a tool)

## Status
- [ ] not started
