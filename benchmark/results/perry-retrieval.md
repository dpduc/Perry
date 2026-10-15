# Perry retrieval only (no chat model)

Share of required source pages among the top 6 hits for the question as asked: hybrid search's own top hits, and the top 6 the reranker picks from a wider pool of hybrid search's hits. The Knowledge Agent may rephrase and search up to 3 times, so this is a lower bound on what it sees.

| Type | Questions | Hybrid search: page recall | Hybrid search: all pages found | Reranked: page recall | Reranked: all pages found |
|---|---|---|---|---|---|
| lookup | 6 | 100% | 6/6 | 100% | 6/6 |
| multi_hop | 6 | 92% | 5/6 | 92% | 5/6 |
| thematic | 5 | 47% | 1/5 | 57% | 1/5 |
| entity | 6 | 92% | 5/6 | 100% | 6/6 |
| all | 23 | 84% | 17/23 | 88% | 18/23 |

| Question | Hybrid search: page recall | Hybrid search: missing pages | Reranked: page recall | Reranked: missing pages |
|---|---|---|---|---|
| L1 | 100% | - | 100% | - |
| L2 | 100% | - | 100% | - |
| L3 | 100% | - | 100% | - |
| L4 | 100% | - | 100% | - |
| L5 | 100% | - | 100% | - |
| L6 | 100% | - | 100% | - |
| M1 | 50% | supplier-handbook.pdf#page=1 | 50% | supplier-handbook.pdf#page=1 |
| M2 | 100% | - | 100% | - |
| M3 | 100% | - | 100% | - |
| M4 | 100% | - | 100% | - |
| M5 | 100% | - | 100% | - |
| M6 | 100% | - | 100% | - |
| T1 | 33% | kestrel-k3-postmortem.pdf#page=3, board-minutes-2025.pdf#page=2 | 33% | kestrel-k3-postmortem.pdf#page=3, board-minutes-2025.pdf#page=2 |
| T2 | 67% | halden-annual-report-2025.pdf#page=3 | 67% | halden-annual-report-2025.pdf#page=3 |
| T3 | 0% | kestrel-k3-postmortem.pdf#page=2, kestrel-k3-postmortem.pdf#page=3 | 50% | kestrel-k3-postmortem.pdf#page=3 |
| T4 | 100% | - | 100% | - |
| T5 | 33% | halden-annual-report-2025.pdf#page=4, board-minutes-2025.pdf#page=2 | 33% | halden-annual-report-2025.pdf#page=4, board-minutes-2025.pdf#page=2 |
| E1 | 50% | kestrel-k3-postmortem.pdf#page=1 | 100% | - |
| E2 | 100% | - | 100% | - |
| E3 | 100% | - | 100% | - |
| E4 | 100% | - | 100% | - |
| E5 | 100% | - | 100% | - |
| E6 | 100% | - | 100% | - |
