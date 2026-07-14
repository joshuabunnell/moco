<!-- source: https://docs.rc.asu.edu/project-storage -->
# Horizon - Project Storage (/data) | ASU RC Docs

Dedicated project-based storage for Phoenix and Sol, mounted at `/data`.

## Cost
| Storage Type | Protocols | Purpose | Free Starting Amount | Annual Cost per TB |
|---|---|---|---|---|
| Project-based Storage | NFS, SMB, Globus | Project file storage | 100GB | $50 |

## Access
Auto-mounted, may not appear in listings until accessed. NFSv3 by default; NFSv4 with ACLs available on request. SMB available campus-wide over Cisco VPN for configured shares.

## Usage guidelines
Must comply with institutional data policy/legal regs; not for sensitive data. Regular backups of critical data still recommended.

## Backup
Nightly snapshots at array level, resilient but not redundant, max 14-day retention, no off-site replication.

## Directory listing
`/data` has thousands of entries; unlisted until first access (efficiency feature) but direct paths still work. Web portal bypasses listing requirement.
