# CALB-Shield Internal Role-Separated Emulation Framework

**Document Status:** INTERNAL DEVELOPMENT & REGRESSION SCAFFOLDING  
**Authoritative Notice:**  
> ### `AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION`

---

## 1. Scope & Research Objective Separation

This framework provides an isolated, software-level role-separated execution harness for internal dry-runs, regression testing, and end-to-end pipeline verification of CALB-Shield.

### Primary Research Objective Preservation
The authoritative scientific goal of CALB-Shield Phase 6 is an **Independent External Red-Team Evaluation ($N=86$)**.
- The registered external evaluation protocol ([`PROPOSED_BLIND_RED_TEAM_EVALUATION_PROTOCOL.md`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/PROPOSED_BLIND_RED_TEAM_EVALUATION_PROTOCOL.md)), external delivery checklist ([`CALB_SHIELD_EXTERNAL_REDTEAM_DELIVERY_CHECKLIST_V2_1.md`](file:///Users/piyush/.gemini/antigravity/brain/444ae7d6-1fcb-4bb6-b1f7-e1ade9c1ce10/CALB_SHIELD_EXTERNAL_REDTEAM_DELIVERY_CHECKLIST_V2_1.md)), registered intake validator ([`benchmarks/validate_redteam_intake.py`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/benchmarks/validate_redteam_intake.py)), and frozen thresholds ([`configs/frozen_thresholds.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/configs/frozen_thresholds.json)) remain **completely untouched and strictly reserved**.
- Phase 6 execution remains **STRICTLY BLOCKED** until genuine external artifacts are delivered.
- This internal emulation workflow operates in a completely separate directory namespace (`internal_emulation/`) and has no authority to validate Phase 6.

---

## 2. Integrity vs. Authenticity: Cryptographic Disclosure

It is essential to distinguish between **data integrity** and **origin authenticity**:
1. **Data Integrity (Enforced via Detached SHA-256 Checksums):**
   - Both the internal vault and execution traces utilize detached SHA-256 checksums (`*.sha256`).
   - This provides **deterministic integrity verification**: it guarantees byte-for-byte immutability across evaluation stages, proving that traces were committed before the vault was opened and that files were not corrupted in transit.
2. **Origin Authenticity (Limitation of Local Emulation):**
   - Because these SHA-256 checksums are generated on the local machine without an asymmetric cryptographic digital signature (e.g., Ed25519 or GPG PKI keypair tied to an independent external identity), they establish **tamper-evidence across sequential steps**, but **not independent provenance or non-repudiation of an external adversary**.
   - True external validation requires cryptographic commitments signed by an external entity prior to Blue Team receipt.

---

## 3. Structural Limitations of Software-Level Role Separation

When role-separated workflows are executed on a single host by the same operator or development team:
- The operator or execution agent possesses physical filesystem and memory access to both the blinded runner and the unsealed vault.
- While the runner CLI strictly disallows access to the vault (enforcing code-level blinding), software-level separation cannot guarantee institutional independence or cognitive blinding.
- **Academic Citation Mandate:** All outputs from this directory are watermarked `AUTOMATED ROLE-SEPARATED EMULATION — NOT INDEPENDENT EXTERNAL VALIDATION` and must be reported as internal role-separated emulation in any scientific publication.

---

## 4. Architecture & Role Decoupling

The emulation pipeline implements three decoupled roles operating on controlled internal fixtures:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        ROLE-SEPARATED EMULATION ARCHITECTURE                           │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ROLE A: INTERNAL FIXTURE PACKAGER (`prepare_internal_fixtures.py`)                    │
│ • Compiles internal development adapters with known provenance.                        │
│ • Generates `emulation_blind_manifest.json` (zero labels or triggers).                 │
│ • Generates `emulation_sealed_vault.json` and detached `emulation_sealed_vault.sha256`.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ GATEKEEPER: INTAKE VALIDATOR (`validate_emulation_intake.py`)                          │
│ • Verifies frozen policy SHA-256 == fec9c02ece0f58181e93df87f8011850...                │
│ • Verifies detached vault checksum, schema whitelists, and absence of leak keys.      │
│ • Enforces path collision protection against external Phase 6 locations.              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ROLE B: BLIND EVALUATION RUNNER (`run_emulation_evaluation.py`)                        │
│ • Accepts `emulation_blind_manifest.json` ONLY (zero vault access).                    │
│ • Executes static admission screening + runtime sidecar generation inspection.         │
│ • Commits traces: `committed_emulation_traces.json` + detached `.sha256`.              │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ROLE C: INTERNAL ADJUDICATOR (`adjudicate_emulation.py`)                               │
│ • Verifies committed trace checksum BEFORE unsealing vault.                           │
│ • Unseals `emulation_sealed_vault.json`, scores against 5-tier rubric.                 │
│ • Computes Balanced Accuracy (MOVER-Wilson CIs), Gate 4 structural, Gate 5 latency.    │
│ • Permanently stamps outputs with the mandatory emulation disclosure.                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Frozen Detection Policy Lock

The internal emulation workflow binds to the immutable detection policy at [`configs/frozen_thresholds.json`](file:///Users/piyush/Desktop/Research%20paper/implementation_2/calb_shield_unified/configs/frozen_thresholds.json):
```
fec9c02ece0f58181e93df87f8011850ee2187bac430ae7f89598ef5ab6fb1fb
```
Every script verifies this digest prior to execution. The harness accepts no command-line overrides for thresholds and fails closed upon any hash discrepancy.

