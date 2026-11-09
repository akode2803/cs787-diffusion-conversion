# Running a CS787 job on your Colab account: setup (about 15 minutes, once)

Thanks for lending CUs. The job is one experiment: the reversal test, run 3 (name-free prompts). It trains two
LoRA adapters on Qwen2.5-7B and tests them.

- **Cost:** about 12 CU, capped at 15 CU. The VM stops itself after 1.5 h at most.
- **Time:** about 70 minutes on one G4, unattended.
- **What it touches:** one Colab session that it creates, named `cs787-runner`. It never touches your other
  sessions. On Drive, it reads and writes only inside the shared `CS787-Diffusion` folder.

After setup, you never run anything yourself. Your AI agent reads requests from this repository and launches the
job. The job reports its results to the repository by itself.

## What you do once

1. **GitHub.** Accept the collaborator invite for this repository (from `akode2803`), then
   `git clone https://github.com/akode2803/cs787-diffusion-conversion.git` and `cd` into it.
2. **Tools.** You need `rclone`, `git`, `python3` and `perl`. On macOS: `brew install rclone`. On Windows, use WSL.
   Then:
   - Install the Colab CLI: `pip install google-colab-cli==0.7.5`.
   - Log in with the Colab account that has the CUs, as `colab readme` describes.
   - Check it works: `colab usage` should print your balance.
3. **Drive.** Aryan has shared the `CS787-Diffusion` folder with you. Open it in the browser and copy the folder
   ID, the last part of the URL (`https://drive.google.com/drive/folders/<THIS PART>`). Then run:
   ```bash
   rclone config create cs787shared drive scope=drive root_folder_id=<FOLDER_ID> --config ~/.config/rclone/cs787.conf
   ```
   A browser window opens for your Google login. Then check it:
   ```bash
   rclone --config ~/.config/rclone/cs787.conf lsf cs787shared:runs/qwen2.5-7b/byod-retrained/best
   ```
   It should list `adapter_config.json`.

   This config file holds your Drive login and gets uploaded to your own Colab VM. Keep only this one remote in
   it, and never share or commit it.
4. **Settings.** Run `cp runner/local.env.example runner/local.env`. The defaults match the steps above.
5. **Agent.** Open your AI agent (Claude Code, Codex, …) in the repository folder and give it this prompt, once:

   > Read runner/AGENT.md and do its One-time setup. My CU budget for this project is 20. From now on, follow
   > its "Every prompt" section at the start of every prompt in this repository.

   It checks everything, makes the protocol load automatically from then on, and starts any job that is waiting.

## After that

Prompt your agent in the repository any time ("check the channel" is enough). It picks up new requests and
reports finished jobs.

If your laptop sleeps during a job, nothing is lost:
- The VM stops itself.
- The agent posts the result the next time you prompt it.

## If something goes wrong

- **Out-of-budget question:** the agent asks you before spending more than the budget.
- **Stopping the VM yourself:** `colab stop -s cs787-runner`. Do this only for that session.
- **`CHECK FAILED: cannot write`:** ask Aryan to make you Editor on the folder.
- **`CHECK FAILED: git push`:** accept the invite.
- **`CHECK FAILED: balance`:** the account has fewer than 15 CU. Tell Aryan.
