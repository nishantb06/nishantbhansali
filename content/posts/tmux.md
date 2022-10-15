---
title: "TMUX for Machine Learning Engineers"
date: 2022-10-16T00:06:25+05:30
draft: false
---

## What is Tmux

TMUX (Terminal Multiplexer) is a program which helps create and manage various terminal sessions created from a terminal itself. We can detach these newly created terminal which helps in asyncronously running multiple programs. 

These terminal will keep on executing a particular command in the background untill we explicitly stop it after attaching it to an active terminal session.

We can create multiple terminal sessions and view and manage them in the same window by toggling between them. Each of these sessions can be detached similary. 

## How ML Engineers use TMUX

As ML Engineers we often extensive training on cloud services like AWS or GCP. To do so we ssh into the vm and giving us a terminal through which we run our training jobs. 

Often training a heavy ML model takes days. During this time if the terminal is disconnected from the VM due to inconsistent network connection or  simply even if the pc goes into sleep mode the terminal will disconnect resulting in abrubplty stopping of our trainig job. 

This is why it is better to create a tmux terminal from our terminal, run the training job on this new terminal and detach it. Now even if the terminal is disconnected from the VM the training job wont stop unless either we stop the VM itsself or in case of internet issues. 

We can also use terminal windows to manage or teminal commands better. For example we can use a new terminal window for each of these jobs

1. Git commands
2. Docker commands
3. To monitor all different processes. `htop -i`
4. To monitor all the GPU’s through `nvdia-smi`
5. General commands like `ls` and `mv` and `pip installs`

## TMUX commands Cheatsheet

Creating and managing terminals . Below commands are for the root terminal

- `sudo apt-get install tmux`  Installing on Linux environment
- `tmux ls` List all active sessions
- `tmux attach -t <name>` Atach a deattached session to the current terminal
- `tmux kill-session -t <name>` Delete a terminal session
- `tmux new -s <name>` Create a new terminal session
- `tmux rename-session -t <name> <new-name>` Rename a terminal session

These commands are for navigating your way through when you are inside an active tmux session

- `Cntrl + b` - This is prefix for any tmux terminal command
- `Cntrl + b` + % Split panes horizontally
- `Cntrl + b` + `—->`, `<-—` switch between right and left panes
- `Cntrl + b` + `[` to enter scrolling mode, `q` to quit
- `Cntrl + b`  + `d` Detach the terminal
- `Cntrl + b`  + `c`  New terminal instance
- `Cntrl + b` + `space` cycle through the terminal instances
- `Cntrl + b` + `p` or `n` previous or next terminal instance
