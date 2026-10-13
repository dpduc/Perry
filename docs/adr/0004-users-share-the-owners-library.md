# Users share the Owner's library; only the Owner acts

Perry has Users who log in with a username and password, but it is still one person's assistant. Every User searches the same Documents and Datasets, which only the Owner uploads. Each User sees only their own Threads. Only the Owner's Turns may involve the Task Agent. Other Users are told that calendar and email aren't available to them.

The Users are a handful of known people (for now the Owner on test accounts, later a supervisor and classmates) looking at the Owner's work. So the library is the point and doesn't need splitting. Threads are private because conversations are personal. The Task Agent is Owner-only because the Workspace is the Owner's own Google account. Letting other Users act on it would let them read the Owner's email and send mail as the Owner. Letting each User connect their own Google account would need a web OAuth flow, and the privacy rule blocks it for any real account while chat runs on a cloud model.

## Considered Options

- **Fully separate Users** (own Documents, Datasets, Threads and Workspace): rejected for now. Same-name uploads would need a per-User scope, and every query would need an owner filter. Moving here later is a migration of `documents`, `chunks`, `datasets` and the ColBERT index.
- **Shared everything, with login only to keep strangers out**: rejected. Every User would see every Thread.

## Consequences

- `threads` and `usage_ledger` record their User. `documents`, `chunks` and `datasets` do not.
- There is one Daily Cap for everyone. Add a per-User cap when people other than the Owner start using Perry.
- Registration is by Invite only, because the app is reachable from a public ngrok URL and every Turn spends the Owner's quota and budget.
