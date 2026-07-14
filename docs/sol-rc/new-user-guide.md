<!-- source: https://docs.rc.asu.edu/new-user-guide -->
# New User Guide | ASU RC Docs

## Getting Started

Welcome to the Research Computing supercomputers! This guide provides quick references and links to get you started smoothly.

This guide assumes you [already have an account](/getting-access) by filling out the request form.

Below is a brief overview of basic tasks to use the supercomputer:

- Connect to the ASU SSLVPN
- Choose how to interact with the supercomputer, e.g., terminal session or web portal
- Transfer files from your workstation
- Run interactive sessions or batch jobs
- Access software resources

## Important Terms

- **Login node / Head node**: A shared node used for simple, non-computational tasks used as the first step onto the supercomputer. It is from this node you request dedicated resources such as CPUs and GPUs for computation and storage-heavy activity.
- **Compute node**: A node in which individual CPUs and GPUs are independently dedicated to users and job allocations. These are suitable for highly-demanding computations and significant changes to your files/storage.
- **Job Scheduler**: The Slurm job scheduler manages all the resources (CPUs/Memory/GPUs/etc.) in the supercomputer and balances their allocation to users based on priority and resource requests.
- **Fairshare**: The prioritization mechanism for the scheduler to decide the order in which jobs run (which differs from the order they were submitted by users). Successful allocation of supercomputing resources such as CPUs, GPUs, and memory is the primary factor that determines the ordering of job start times.

## Connect through Cisco VPN

Using the supercomputer requires continued connectivity to the [ASU SSLVPN via Cisco AnyConnect](/sslvpn). The VPN guarantees connectivity to all Research Computing resources, including the supercomputers, virtual machines, storage backends, and the web portal.

In addition to ensuring connectivity, the VPN also helps maintain stable connections to these services through direct routing and optimizing packet delivery. If you experience any issues with connectivity or service intermittence, ensure you are using the ASU SSLVPN _even if on campus networks_.

## Login to the Supercomputer

Brief description of available connection methods.

- **Web Portal**: The Open-on-Demand [Web Portal](/web-portal) is accommodating for new users and graphical applications.
- **SSH (Terminal)**: [Terminal sessions](/terminal-clients) are recommended for most users and for batch submissions.

## Transfer Files

Your access comes with access to various filesystems with which you can store your data, scripts, and other files required for your work on the supercomputer. This includes your `HOME` directory, your scratch directory, and group-owned shared storage. Read more about [supercomputer storage options](/file-system-overview).

There are multiple methods you can use to transfer files to and from your workstation, recommended in order:

- **[Globus](/transferring-to-supercomputer#globus)**: Accessible through a [web browser](https://globus.org) or a desktop application, this is the #1 recommended way to move large files. It features a simple drag-and-drop interface which will move files, validate their replication, and provide notifications of completion. This is the also the most speedy method to move files.
- **[SCP](/transferring-to-supercomputer#command-line-tools)**: Often offered as a desktop application (e.g., [Filezilla](https://filezilla-project.org), [WinSCP](https://winscp.net)), this is a straightforward way to move files on a smaller-scale. It is also possible to move files through a terminal window using the command `scp`.
- **[Web Portal](/transferring-to-supercomputer#web-portal)**: Each supercomputer also provides a [browser-based web interface](/web-portal) to move files, but it comes with limitations: it cannot move large files and it is the slowest of the the available options. Nonetheless, for small transfers and for navigating your file structure, it is easy-to-use.

## Run Interactive Sessions or Create SBATCH Scripts

You can use compute resources interactively or non-interactively through what is called _batched job submissions_. Read a brief overviews of [running interactive sessions](/interactive-sessions) or [creating SBATCH scripts](/slurm-sbatch).

---

## Navigation

[Previous: About Research Computing](/about) | [Next: Get Help](/contact-us)

## Table of Contents

- [Important Terms](#important-terms)
- [Connect through Cisco VPN](#connect-through-cisco-vpn)
- [Login to the Supercomputer](#login-to-the-supercomputer)
- [Transfer Files](#transfer-files)
- [Run Interactive Sessions or Create SBATCH Scripts](#run-interactive-sessions-or-create-sbatch-scripts)
