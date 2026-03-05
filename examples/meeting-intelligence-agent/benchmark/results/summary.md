# LiquidAI/LFM2.5-1.2B-Thinking-GGUF:Q4_0

**Date:** 2026-03-04 12:34  
**Score: 3/10 tasks passed**

| # | Difficulty | Task | Pass | Time | In tokens | Out tokens | Turns | Tool calls |
|---|---|---|:---:|---:|---:|---:|---:|---|
| 1 | easy | Read transcript and list attendees | ✓ | 3.0s | 4,361 | 125 | 3 | 2× `read_transcript` |
| 2 | easy | Look up one team member | ✓ | 1.5s | 1,401 | 79 | 2 | `lookup_team_member` |
| 3 | easy | Create one explicit task | ✓ | 2.1s | 1,500 | 135 | 2 | `create_task` |
| 4 | medium | Look up three team members | ✗ | 1.9s | 1,419 | 124 | 2 | `lookup_team_member` |
| 5 | medium | Create three tasks from a given list | ✗ | 2.1s | 1,642 | 124 | 2 | `create_task` |
| 6 | medium | Read transcript and save a structured summary | ✗ | 2.0s | 2,147 | 113 | 2 | `read_transcript` |
| 7 | hard | Full pipeline: tasks + summary + email | ✗ | 2.1s | 2,157 | 114 | 2 | `read_transcript` |
| 8 | hard | Detect and flag unassigned action item | ✗ | 2.1s | 2,180 | 106 | 2 | `read_transcript` |
| 9 | hard | Default due dates for items without explicit deadlines | ✗ | 2.5s | 2,179 | 148 | 2 | `read_transcript` |
| 10 | hard | Full pipeline: custom filename and targeted email recipients | ✗ | 2.0s | 2,189 | 104 | 2 | `read_transcript` |

**Totals:** 21,175 input tokens · 1,172 output tokens

---

# LiquidAI/LFM2-8B-A1B-GGUF:Q4_0

**Date:** 2026-03-04 12:42  
**Score: 0/10 tasks passed**

| # | Difficulty | Task | Pass | Time | In tokens | Out tokens | Turns | Tool calls |
|---|---|---|:---:|---:|---:|---:|---:|---|
| 1 | easy | Read transcript and list attendees | ✗ | 2.3s | 2,620 | 43 | 2 | — |
| 2 | easy | Look up one team member | ✗ | 1.5s | 2,612 | 49 | 2 | — |
| 3 | easy | Create one explicit task | ✗ | 2.8s | 2,727 | 120 | 2 | — |
| 4 | medium | Look up three team members | ✗ | 2.4s | 2,651 | 104 | 2 | — |
| 5 | medium | Create three tasks from a given list | ✗ | 8.6s | 2,992 | 458 | 2 | — |
| 6 | medium | Read transcript and save a structured summary | ✗ | 3.8s | 2,832 | 181 | 2 | — |
| 7 | hard | Full pipeline: tasks + summary + email | ✗ | 17.0s | 3,228 | 1,001 | 2 | — |
| 8 | hard | Detect and flag unassigned action item | ✗ | 13.3s | 3,081 | 822 | 2 | — |
| 9 | hard | Default due dates for items without explicit deadlines | ✗ | 5.5s | 2,889 | 314 | 2 | — |
| 10 | hard | Full pipeline: custom filename and targeted email recipients | ✗ | 7.3s | 2,982 | 404 | 2 | — |

**Totals:** 28,614 input tokens · 3,496 output tokens

---

# LiquidAI/LFM2.5-1.2B-Instruct-GGUF:Q4_0

**Date:** 2026-03-04 12:54  
**Score: 4/10 tasks passed**

| # | Difficulty | Task | Pass | Time | In tokens | Out tokens | Turns | Tool calls |
|---|---|---|:---:|---:|---:|---:|---:|---|
| 1 | easy | Read transcript and list attendees | ✗ | 2.9s | 4,357 | 107 | 3 | 2× `read_transcript` |
| 2 | easy | Look up one team member | ✗ | 32.6s | 60,629 | 1,726 | 30 | 4× `lookup_team_member`, 7× `save_summary`, 6× `send_email`, 8× `create_task`, 5× `read_transcript` |
| 3 | easy | Create one explicit task | ✓ | 2.0s | 1,500 | 121 | 2 | `create_task` |
| 4 | medium | Look up three team members | ✓ | 28.3s | 48,412 | 1,598 | 24 | 4× `lookup_team_member`, 5× `save_summary`, 6× `create_task`, 4× `send_email`, 4× `read_transcript` |
| 5 | medium | Create three tasks from a given list | ✓ | 7.3s | 5,310 | 438 | 5 | 3× `create_task`, `save_summary` |
| 6 | medium | Read transcript and save a structured summary | ✓ | 82.8s | 168,838 | 4,459 | 30 | 9× `read_transcript`, 8× `save_summary`, 6× `send_email`, 2× `create_task`, 5× `lookup_team_member` |
| 7 | hard | Full pipeline: tasks + summary + email | ✗ | 76.5s | 167,546 | 4,151 | 30 | 11× `read_transcript`, `lookup_team_member`, `create_task`, 9× `save_summary`, 8× `send_email` |
| 8 | hard | Detect and flag unassigned action item | ✗ | 3.6s | 3,831 | 194 | 3 | `read_transcript`, `create_task` |
| 9 | hard | Default due dates for items without explicit deadlines | ✗ | 6.7s | 5,791 | 399 | 4 | `read_transcript`, `create_task`, `save_summary` |
| 10 | hard | Full pipeline: custom filename and targeted email recipients | ✗ | 3.8s | 3,862 | 196 | 3 | `read_transcript`, `create_task` |

**Totals:** 470,076 input tokens · 13,389 output tokens

---

# LiquidAI/LFM2-24B-A2B-GGUF:Q4_0

**Date:** 2026-03-04 13:01  
**Score: 9/10 tasks passed**

| # | Difficulty | Task | Pass | Time | In tokens | Out tokens | Turns | Tool calls |
|---|---|---|:---:|---:|---:|---:|---:|---|
| 1 | easy | Read transcript and list attendees | ✓ | 6.9s | 2,094 | 88 | 2 | `read_transcript` |
| 2 | easy | Look up one team member | ✓ | 4.4s | 1,401 | 59 | 2 | `lookup_team_member` |
| 3 | easy | Create one explicit task | ✓ | 5.0s | 1,495 | 105 | 2 | `create_task` |
| 4 | medium | Look up three team members | ✓ | 27.0s | 9,190 | 601 | 8 | 3× `lookup_team_member`, `save_summary`, `send_email`, 2× `create_task` |
| 5 | medium | Create three tasks from a given list | ✓ | 42.4s | 9,471 | 1,270 | 7 | 3× `create_task`, `save_summary`, 2× `send_email` |
| 6 | medium | Read transcript and save a structured summary | ✓ | 31.4s | 6,536 | 884 | 4 | `read_transcript`, `save_summary`, `send_email` |
| 7 | hard | Full pipeline: tasks + summary + email | ✓ | 64.5s | 30,189 | 1,457 | 14 | `read_transcript`, 3× `lookup_team_member`, 6× `create_task`, `save_summary`, `send_email` |
| 8 | hard | Detect and flag unassigned action item | ✓ | 50.8s | 17,896 | 1,238 | 9 | `read_transcript`, 5× `create_task`, `save_summary`, `send_email` |
| 9 | hard | Default due dates for items without explicit deadlines | ✗ | 32.8s | 8,458 | 821 | 5 | `read_transcript`, `create_task`, `send_email`, `save_summary` |
| 10 | hard | Full pipeline: custom filename and targeted email recipients | ✓ | 103.0s | 34,575 | 2,897 | 13 | `read_transcript`, 6× `create_task`, 2× `send_email`, 3× `save_summary` |

**Totals:** 121,305 input tokens · 9,420 output tokens
