<!-- source: https://docs.rc.asu.edu/sharing-files -->
# Sharing Files with Other Users | ASU RC Docs

For sharing outside your own group (within-group: use group-owned `/data` shares instead).

## Process
1. Recipient creates a world-writable directory.
2. Sender copies files into it.
3. Recipient revokes writable permissions.
4. Recipient now owns their copy.

### 1. Connect to the data transfer node (services both Sol and Phoenix)
```
ssh <asurite>@soldtn.sol.rc.asu.edu
```

### 2. Recipient creates receiving directory
**Sol:**
```
chmod -R o+rx /scratch/<recipient>
install -d -m 777 /scratch/<recipient>/receiving_dir
```
**Phoenix:** same but `/phxscratch/<recipient>/...` (data still physically in `/scratch` — `/phxscratch` is only a transfer-node path alias for cross-cluster transfers).

**Warning: never do this with `/home` directories** — such changes get reverted without notice.

### 3. Sender copies files
```
cp -R /path/to/senders/file/or/files /scratch/<recipient>/receiving_dir
```

### 4. Recipient revokes permissions
```
chmod o-rwx /scratch/<recipient>/
```

For frequent/ongoing collaboration, purchased [project storage](/project-storage) is more sustainable than this manual process.
