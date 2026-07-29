# Project instructions

- When you add anything undocumented (a new module, script, config option, data file, or behavior change) that isn't self-evident from the code, add a dated entry to [CHANGELOG.md](CHANGELOG.md). Keep entries to the *why*, not a restatement of the diff.
- If significant changes or discoveries have been made on a topic (e.g. a batch of rare coins identified, a dataset scraped, a model's behavior characterized), write them up in `docs/` as a topic file (e.g. `docs/rare-coins.md`), not a one-off root-level file. Before creating a new doc, check `docs/` for an existing file on the same or a similar topic and update/extend that one instead of starting a new one. Link to the relevant doc from the changelog entry rather than expanding inline there.
