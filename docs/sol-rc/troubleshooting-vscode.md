<!-- source: https://docs.rc.asu.edu/troubleshooting-vscode -->
# Troubleshooting Visual Studio Code

## Web Portal Tunnel Erroring out

If your VSCode tunnel job initiates but the tunnel itself fails, you may encounter an error message indicating VPN unavailability.

### Root Causes
- Multiple tunnels running simultaneously
- A tunnel that did not shut down properly

VSCode enforces a single tunnel instance through a lock file mechanism. If a tunnel already exists, starting another will fail. Similarly, an unclean shutdown leaves the lock file in place, preventing new tunnel creation.

### Solution
```bash
rm -rf ~/.vscode
find ~/.vscode-server/data/ -mindepth 1 -maxdepth 1 -type d ! -name "Machine" -exec rm -rf {} +
```
Launch a new job afterward, and the tunnel will initialize properly.

## Repeated DUO Authentication Attempts

### Issue Description
When connecting VSCode to the supercomputer, every action—terminal login, file uploads, file saves—may trigger repeated password and DUO authentication requests.

### Cause
**This occurs when connecting directly to login nodes via SSH**, as these nodes require DUO authentication for security.

### Resolution
**Create a VSCode tunnel through the web portal instead.** This establishes a single connection permitting VSCode to execute multiple commands without interruption from authentication prompts.
