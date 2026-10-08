# Perry

A personal assistant that answers from your own documents and data, and acts on your calendar and email, through one conversation.

## Language

### Orchestration

**Supervisor**:
The coordinator that reads a request, decides which Agents to involve and in what order, and composes the final answer.
_Avoid_: Router, orchestrator, master agent

**Agent**:
A specialist with one area of responsibility that the Supervisor delegates to. Perry has exactly three: the Knowledge Agent, the Analysis Agent and the Task Agent.
_Avoid_: Worker, sub-bot, skill

**Knowledge Agent**:
The Agent that finds relevant Chunks in your documents and answers with Citations.
_Avoid_: Retrieval agent, RAG agent, search agent

**Analysis Agent**:
The Agent that answers questions by querying and computing over Datasets.
_Avoid_: Data agent, SQL agent

**Task Agent**:
The Agent that reads from and acts on your calendar and email.
_Avoid_: Action agent, execution agent

**Turn**:
One request from you and everything Perry does to answer it, possibly involving several Agents in sequence.

**Thread**:
A resumable conversation made of Turns.
_Avoid_: Session, chat, conversation history

### Knowledge

**Document**:
A file you have uploaded so Perry can search it. A newer upload with the same name replaces the older one.
_Avoid_: File, source, attachment

**Chunk**:
A passage of a document that can be retrieved on its own and traced back to its page.
_Avoid_: Snippet, passage, fragment

**Citation**:
A pointer from an answer to the exact document page a claim came from.
_Avoid_: Source, reference

**Dataset**:
A table of your structured data (uploaded from CSV or a spreadsheet) that the Analysis Agent can query.
_Avoid_: Table (when talking to users), spreadsheet

### Actions

**Read Action**:
An action on calendar or email that only looks, such as listing events or fetching messages. Runs without asking.

**Write Action**:
An action on calendar or email that changes something or reaches another person, such as sending, scheduling or deleting.
_Avoid_: Side effect, mutation

**Confirmation**:
Your explicit approval of one specific Write Action, shown exactly as it will be performed and editable before you approve. No Write Action happens without one, and each one covers a single attempt.
_Avoid_: Approval prompt, consent
