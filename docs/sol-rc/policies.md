<!-- source: https://cores.research.asu.edu/research-computing/policies -->
# Research Computing Acceptable Use Policy

## Purpose
Furthers scientific/research/educational efforts at ASU and partners. RC staff may modify or cancel any activity that interferes with this purpose.

## 1. University Policies
All users must follow ASU's "Computer, Internet, and Electronic Communications Information Management Policy" (ACD125) plus departmental/college requirements.

## 2. Communications
Outage/maintenance emails go to ASURITE accounts. Unsubscribing locks the account (and any accounts it sponsors) until re-enrolled.

## 3. University Use Only
Resources are for research/scholarly work only. **"Any use of the supercomputer for personal or commercial gain is strictly prohibited."**

## 4. Maintenance
Scheduled maintenance windows (governance-board approved); emergency corrective action can happen outside them.

## 5. Restricted Data Sets
Complies with ASU Data Handling Standards Levels 1-2. CUI/ECI/HIPAA/FERPA/PII/PHI or other regulated data may need Level 3/4 — consult RC staff before use.

## 6. Licensing
Users responsible for licensing of self-installed software.

## 7. Cryptocurrencies / Cryptographic Mining / Distributed Computing
Mining, distributed cryptography (e.g. distributed.net), and volunteer computing (e.g. SETI@Home) are **prohibited** without specific prior written RC consent.

## 8. Project and Long-term Storage
RC can enforce quotas and delete/move/restrict files on long-term (Canyon) and project (`/data`) storage as needed. Long-term storage isn't directly compute-accessible — must transfer out first. No guarantee against data loss; project storage has hardware redundancy + 14-day snapshots but no cross-system copies; long-term storage has no redundancy/snapshots/replication at all.

## 9-10. Co-located Hardware / Virtual Systems
(Not relevant to Sol supercomputer usage — paid infrastructure/VM hosting services with their own security, payment, and warranty terms.)

## 11. Supercomputers

### 11.1 Access
Available to ASU faculty/staff/students/affiliates; non-eligible applicants need a sponsor.

#### 11.1.1 Course Allocations
Instructors requesting class allocations agree to the AUP on students' behalf and are responsible for ensuring students understand/follow it.

### 11.2 Jobs
Violations can mean lockouts; severe/repeated ones mean full access removal.

#### 11.2.1 Login Node Restrictions
**"Users should run jobs on the supercomputer compute nodes configured for this purpose. Running jobs on the supercomputer's login nodes is prohibited and will generate a warning."** RC staff may terminate such jobs without notice.

#### 11.2.2 Performance Impact
Any job/process degrading other users' jobs → terminated without advance notice.

#### 11.2.3 Excessive Idleness
**Idle interactive sessions for 4+ hours, or "sleep loops," → terminated without advance notice.** (Relevant if Claude Code or a VSCode tunnel is left inside an `interactive`/`salloc`/`lightwork` allocation without real activity for hours — this rule targets compute-node interactive sessions, not the login node itself, but is a real risk if we ever move to running inside an allocated session.)

#### 11.2.4 Unattended Listening Services
Jobs that open listening ports for other machines to connect to → terminated without advance notice. (VSCode's tunnel/web-portal flow is RC's own sanctioned mechanism for remote access, so it isn't what this targets — this is aimed at ad-hoc port-forwarding/servers inside jobs.)

### 11.3 Scratch File System Use
Temporary/shared, for active computation, not suitable as home-directory replacement.

#### 11.3.1 Active Use Definition
Access/modification within 90 days, or designated as a reference dataset for jobs. Notifications precede removal. Sponsors are responsible for migrating a deactivated user's data. Each user must limit usage to avoid impacting others.

#### 11.3.2 Scratch Storage Limits
Per-user limits apply; **max 20 million files per user**. RC may temporarily expand limits case-by-case via RC Leadership consultation.

### 11.4 Home Directories
**100GB limit**, intended for software installs and package environments (e.g. Python envs). Persistent — not purged like scratch.

### Data Protection (Supercomputers)
No guarantee against loss/corruption. Scratch: no redundancy/snapshots/replication at all. Home: hardware redundancy + 14-day change snapshots, no cross-system copies.

## 12. Violations
RC may lock or permanently remove accounts (including supercomputer/server/storage access) for policy violations, at RC's discretion.

## 13. Policy Updates
RC may amend policy at any time and publish changes to the RC website.

## Policy History
| Date | Update |
|------|--------|
| 4/15/2019 | Initial Release |
| 4/19/2019 | Combined AUP and Scratch Policies |
| 08/31/2023 | Updated Sol Scratch Policy |
| 11/01/2023 | Added Communications Policy |
| 2/13/2025 | Updated available RC Services (VMs, project/long-term storage, co-location) |
