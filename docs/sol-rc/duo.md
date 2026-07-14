<!-- source: https://docs.rc.asu.edu/duo -->
# Duo Two-Factor Authentication (2FA) - ASU RC Docs

## Overview

"Starting **January 5th, 2026**, all users accessing ASU Research Computing resources must utilize Duo Two-Factor Authentication (2FA)." This security requirement protects accounts and supercomputing infrastructure from unauthorized access.

## SSH Access with Duo 2FA

When connecting via SSH, users enter ASURITE credentials and then receive a Duo Push notification on their registered 2FA device for approval.

**Key points:**
- Multiple registered devices: notifications go to the default device
- Manage devices via the [Duo Device Management Portal](https://weblogin.asu.edu/2fa/selfservice/device-management)
- Data transfer tools (scp, rsync, CyberDuck, FileZilla, WinSCP) require Duo 2FA
- Duo Phone and SMS unavailable for SSH access

**Exception:** "SSH access via pre-configured public/private key pairs does not require Duo 2FA. Workloads that use key-based authentication, such as automated scripts or batch jobs, can continue to do so without modification."

## Web Portal Access with Duo 2FA

Logging into the web portal requires Duo 2FA through ASU Single-Sign-On (SSO). Web-based terminal connections from within the portal do not require additional Duo verification.

## Support Resources

Duo is managed by ASU's Enterprise Technology department:
- [ASU Duo Support Page](https://getprotected.asu.edu/services/identity-and-access-management/duo-two-factor-authentication/gethelp)
- [ASU Experience Center support](https://tech.asu.edu/services/ec)

Users with workflow concerns should [contact Research Computing](/contact-us).
