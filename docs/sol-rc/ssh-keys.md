<!-- source: https://docs.rc.asu.edu/ssh-keys -->
# Passwordless SSH Keys | ASU RC Docs

## SSH Keys

SSH keys enable cryptographically-secure authentication using a keyfile instead of passwords. SSH keys permit immediate login to Sol without Duo authentication.

## Using SSH Keys

### Generate an SSH Key

```
$ ssh-keygen -t ed25519
Generating public/private ed25519 key pair.
Enter file in which to save the key (/home/user/.ssh/id_ed25519):
Enter passphrase for "/home/user/.ssh/id_ed25519" (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in /home/user/.ssh/id_ed25519
Your public key has been saved in /home/user/.ssh/id_ed25519.pub
```

You may choose to use or not use a passphrase as a password lock on key usage.

**Note:** "This passphrase does not and should not correspond to your ASURITE password." Using this key eliminates the need for your ASURITE password during login.

### Copy the Public SSH Key

Use `ssh-copy-id` to copy the public key (.pub) to your target host. Maintain your private key on your personal workstation and safeguard it—the private key is the only sensitive component.

```
$ ssh-copy-id rcsparky@sol.asu.edu
```
(Still requires Duo approval + password for this one-time copy step.)

#### Multiple Keys on Your System
```
ssh-copy-id -i ~/.ssh/id_ed25519.pub asurite@sol.asu.edu
```

### Login with the SSH Key

Test the connection—it should log you in without requesting your ASURITE password or Duo authentication:
```
ssh asurite@sol.asu.edu
```
or
```
ssh -i ~/.ssh/id_ed25519 asurite@sol.asu.edu
```
