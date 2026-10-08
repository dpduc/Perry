# Perry retrieval only (no chat model)

Share of required source pages among the top 6 hybrid-search hits for the question as asked. The Knowledge Agent may rephrase and search up to 3 times, so this is a lower bound on what it sees.

| Type | Questions | Page recall | All pages found |
|---|---|---|---|
| lookup | 6 | 83% | 5/6 |
| multi_hop | 6 | 92% | 5/6 |
| thematic | 5 | 67% | 2/5 |
| entity | 6 | 83% | 4/6 |
| all | 23 | 82% | 16/23 |

| Question | Page recall | Missing pages |
|---|---|---|
| L1 | 100% | - |
| L2 | 0% | halden-annual-report-2025.pdf#page=3 |
| L3 | 100% | - |
| L4 | 100% | - |
| L5 | 100% | - |
| L6 | 100% | - |
| M1 | 50% | supplier-handbook.pdf#page=1 |
| M2 | 100% | - |
| M3 | 100% | - |
| M4 | 100% | - |
| M5 | 100% | - |
| M6 | 100% | - |
| T1 | 33% | kestrel-k3-postmortem.pdf#page=3, board-minutes-2025.pdf#page=2 |
| T2 | 67% | halden-annual-report-2025.pdf#page=3 |
| T3 | 100% | - |
| T4 | 100% | - |
| T5 | 33% | halden-annual-report-2025.pdf#page=4, board-minutes-2025.pdf#page=2 |
| E1 | 50% | kestrel-k3-postmortem.pdf#page=1 |
| E2 | 100% | - |
| E3 | 50% | halden-newsletter-autumn-2025.pdf#page=2 |
| E4 | 100% | - |
| E5 | 100% | - |
| E6 | 100% | - |
