<!-- source: https://docs.rc.asu.edu/resource-limits -->
# Resource Limits | ASU RC Docs

## Storage Resource Limits

- **Home directory**: quota of **100GiB** per user.
- **Scratch directory** (`/scratch/[asurite]`): no fixed quota, but publicly-shared — time limits on unused data apply (must be actively used for computation or moved to long-term storage). (Note: /scratch banner we saw states a de facto 100TB/user ceiling with 90-day purge on unused files.)
- **Project Storage**: 100GiB shared folder per faculty sponsor/PI, accessible to PI and all sponsored students/scholars/affiliates.
- **Class Storage**: 100GiB shared folder per class at `/data/courses/<year>/<class>`, accessible to instructor + students.

## Standard Account Limits
- Max **2 jobs queued** in the `debug` queue.

## Class Account Limits
- Max **2 jobs running concurrently** per user
- Max **10 jobs in the queue** per user
- Max **960 GPU running minutes** per user (1 GPU/16h, 2 GPUs/12h, 3 GPUs/8h, or 4 GPUs/4h)
- Max **32 CPU cores, 320GB memory, 4 GPUs** per job
- Max **wall time 24 hours** per job

Users with both Academic Course and Research accounts must specify `-A class_...` or `-A grp_...` to pick which account/limits apply.
