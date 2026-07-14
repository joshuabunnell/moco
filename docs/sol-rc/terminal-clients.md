<!-- source: https://docs.rc.asu.edu/terminal-clients -->
# SSH Clients | ASU RC Docs

## Using a Terminal Client

To interact with the supercomputer, users typically employ a terminal interface through a command-line or SSH client installed on their local workstation. Once connected, you can submit unattended jobs--tasks executed autonomously on the supercomputer without the need for continuous user interaction or maintaining an active session.

## Examples

### Connect via SSH

**Sol**
```
ssh -X asurite@sol.asu.edu
```

**Phoenix**
```
ssh -X asurite@phoenix.rc.asu.edu
```

> Replace `asurite` with your own ASURITE username. Use `-X` to allow windowed (point-and-click) applications to open on your workstation desktop.

> You will be prompted for a Duo Push notification when connecting via SSH. Make sure to have your 2FA device ready.

## Recommended Terminal Clients

### PuTTY (Windows)
Straightforward and lightweight. Use with ASURITE login/password, hostname, port 22. Can save named sessions.

### Powershell (Windows 11)
Bundled with the OS, usable with the standard connection command.

### WSL: Windows Subsystem for Linux (Windows)
"The ssh client that is included with the Windows Subsystem for Linux **does not work with the Cisco VPN Client** and is therefore unsupported."

### Terminal (MacOS)
Built-in SSH client works fine.

### Linux
Virtually every distribution's included SSH client works without issue.
