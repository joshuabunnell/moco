<!-- source: https://docs.rc.asu.edu/scratch -->
# Scratch File System (/scratch) | ASU RC Docs

Shared storage resource for temporary files, available to all users, for short-term storage during computational jobs. Each user responsible for limiting usage.

## Policy
- **No backup**: not backed up, RC not responsible for loss/deletion.
- **Storage limit**: 100 TB per user maximum.
- **Extension requests**: possible under extenuating circumstances, case-by-case, no guarantee.
- **File retention**: files not accessed for 90 days are removed.
- **Notification**: RC notifies file owners, sponsoring PIs, and departments before deletion.
- **Long-term storage**: move anything needing >90-day retention to `/home`, purchased `/data` project storage, or other media.
- **Policy exceptions**: decided by the Research Computing Governance Board.

## Aging/Unused Files (tiered notifications)
1. **45-day warning**: weekly emails start at 45 days of non-access, giving 45 more days to use/move/request extension.
2. **Final warning**: sent <7 days before permanent deletion.

Aging file lists are generated in users' home directories for review.
