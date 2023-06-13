# Setting Up Remote GPU Access

Autoscoper uses GPU for tracking, and usually remote desktop applications (e.g., [Citrix](https://www.citrix.com/solutions/digital-workspace/)) do not allow GPU access when you’re using their service

Currently, there are three ways to work with Autoscoper remotely:

## 1. Changing your Remote PC's Group Policy to Enable GPU Access:

When using remote desktop, Widows bypasses the onboard graphics and creates a virtual graphics driver.
You can get around this using the [group policy setting below](https://www.varonis.com/blog/group-policy-editor):

a) Use your Remote Desktop application and login to your Remote PC

b) Open Search in the Toolbar and type Run or select Run from your Start Menu.

c) Type ‘gpedit.msc’ in the Run command and click OK.

d) Now select these folders in order:

    i. Computer Configuration

    ii. Administrative Templates

    iii. Windows Components

    iv. Remote Desktop Services

    v. Remote Desktop Session Host

    vi. Remote Session Environment

![gpedit](https://github.com/BrownBiomechanics/Autoscoper/releases/download/docs-resources/gpedit-personalization.jpg)

e) Right-click on “Use hardware graphics adapters for all Remote Desktop Services sessions”, and “Enable”
the status of this system.

f) Close the Editor window and open Autoscoper. If it didn’t work, restart your PC, and try again.

g) *Voila!*

## 2. Using High-Performace Computing (HPC) Servers:

If you are using Autoscoper for research, your university probably is providing you with free exploratory
accounts [Brown University: [Center for Computation & Visualization](https://ccv.brown.edu/)]. If you have access to such a system,
you need to use your [Virtual Network Computing (VNC)](https://docs.ccv.brown.edu/oscar/connecting-to-oscar/open-ondemand/desktop-app-vnc) nodes to access the Linux OS. You need to compile
the Autoscoper code on your system. You can find compiling instruction [here](./developer-guide/building-autoscoper.md)

## 3. Open Autoscoper when Remote Session is not connected:

There is another approach, but it's a little bit complicated and needs using [Command-Prompt](https://www.bleepingcomputer.com/tutorials/windows-command-prompt-introduction/). Basically, you
need to open the software while you are not remotely connected to your PC. So, we want to automatically
disconnect the “Remote Session” and open Autoscoper. After we run this code, we remote back into the
system. To do so, you need to create a new text document on your **C:\ drive**, and add these three lines in it:

```batch
tscon 3 /dest:console
cd path\to\autoscoper\executable
autoscoper.exe
```
and save it as **run_autoscoper.bat**. You are changing the filetype here to **.bat**, because you need to use
command prompt to execute this file. Make sure your text file is saved as **.bat** and not **.txt**.
In the code above, you need to change the following sections:

In `tscon 3 /dest:console`, you need to change the number `3`

* You need to change this number to your “Remote Session ID”. You can get this ID by running “query
session” command in CMD (command prompt).

```
SESSIONNAME     USERNAME        ID  STATE   TYPE    DEVICE
console         Administrator   0   active  wdcon
rdp-tcp#1       BardiyaAk       3   active wdtshare
rdp-tcp                         2   listen wdtshare
                                4   idle
```
Now that the script is written, open the command prompt (type **cmd** in Start-Run), go to your C-drive and
type **run_autoscoper.bat**. If the script is written correctly, you get disconnected from your remote session.
Log back in and Autoscoper should be open now.
