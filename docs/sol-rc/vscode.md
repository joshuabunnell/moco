<!-- source: https://docs.rc.asu.edu/vscode -->
# VSCode | ASU RC Docs

## Visual Studio Code IDE

This guide assists users in setting up VSCode for remote tunneling into supercomputers using either a local client or web browser.

> **"These instructions will guide you to create a VSCode tunnel rather than connect via Remote SSH. Remote SSH is discouraged and when used on login nodes can often result in usage violations due to CPU core overuse."**

### VSCode Setup

VSCode functions for managing remote files on supercomputer storage and providing a rich development environment. The setup leverages Visual Studio Code CLI Server. Three configuration methods are available:

**Required Extension:** The Remote - Tunnels extension must be installed to connect with VSCode desktop. ASU Supercomputers use GitHub Authentication.

#### Method 1: Web Portal Setup
1. Navigate to web portal (sol.asu.edu or phx.rc.asu.edu)
2. Select "VSCode Server" under Interactive Apps
3. Complete job options and click "Launch"
4. When GitHub Code appears, select "Copy Code and Login To GitHub"
5. Sign in with GitHub using the automatically copied code
6. Return to web portal; within 20-30 seconds, launch buttons appear for browser or desktop access

#### Method 2: Command-Line Configuration
1. Log into supercomputer and execute `vscode` command:
   ```
   [rcsparky@login02:~]$ vscode
   ```
2. The command accepts same arguments as `interactive` (e.g., `vscode -t 1-0` for 1-day allocation)
3. Note the GitHub login URL and accompanying code from output
4. Terminal provides connection guidance including the tunnel link

**Connecting via Remote Tunnels:**
1. Verify Remote - Tunnels extension is installed
2. Press `F1` and select "Remote-Tunnels: Connect to Tunnel..."
3. If prompted, select "GitHub" authentication
4. Select tunnel name (typically supercomputer name + ASURITE)

The connected tunnel name appears in IDE bottom left corner.

#### Method 3: Open-Source Code Server
1. Navigate to web portal (sol.asu.edu or phx.rc.asu.edu)
2. Select "VSCode Server" under Interactive Apps
3. Complete job options and click "Launch"
4. Once job starts, click "Connect to VSCode Server"
5. Begin coding in the web-based interface

> "Please note that this is the open-source version of VSCode, and does not support the Microsoft Extension Store, so some Microsoft Extensions, such as Co-Pilot, are not available. To use Co-Pilot, you will need to use a VSCode Tunnel."

### Logging In With GitHub

> "The XXXX-XXXX code will be different on every invocation. This code connects your background VSCode daemon to github.com, which provides half the tunneling functionality; the other half is connecting your browser or VSCode IDE to the other end of the tunnel."

**Authentication Process:**
1. Open the provided URL in your workstation browser
2. Log into GitHub account (required)
3. Enter the provided device code when prompted
4. Authorize GitHub with the green button
5. Connection completes automatically
6. Terminal users see IDE connection guidance; portal users see Open in VSCode Web/Desktop buttons within approximately one minute
