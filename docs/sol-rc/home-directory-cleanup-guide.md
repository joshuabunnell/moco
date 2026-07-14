<!-- source: https://docs.rc.asu.edu/home-directory-cleanup-guide -->
# /home Directory Cleanup Guide | ASU RC Docs

100 GiB limit on `/home`. Over quota → `Error: No space left on devices` during jobs/terminal, and web portal session creation becomes unavailable. **SSH access remains unaffected**, so it's the way to fix things even when over quota.

## Finding large files
```bash
gdu ~   # d to delete, q to quit
ncdu ~  # equivalent
```

## Common space consumers
**Mamba**: temp files/cached tarballs/envs in home.
```bash
mamba env remove -n <envName>
mamba clean --all
```

**Pip** (mamba recommended over pip):
```bash
rm -rf ~/.cache/pip
```

**Hugging Face**: downloads models to `~/.cache/huggingface` by default — move to scratch:
```bash
mv ~/.cache/huggingface $SCRATCH
echo "export HF_HOME=$SCRATCH/huggingface" >> ~/.bashrc
source ~/.bashrc
```
