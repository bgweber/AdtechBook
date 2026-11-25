# Contributing

Thanks for helping with *Machine Learning in Mobile AdTech*. Corrections,
exercises, simulator features, and code are all welcome.

## Where to start

- **Corrections.** Typos, broken links, wrong numbers, unclear passages: open
  an issue or send a pull request against the chapter's `.qmd` file in
  `chapters/`.
- **Exercises.** Every "Try it on the simulator" box runs. Part IV's boxes
  also say what scaling up would take: bigger markets, longer histories, and
  wider models. Results from those, written up as a walkthrough section, are
  welcome.
- **The simulator.** The README's "What the simulator leaves out" section
  lists features it doesn't have yet. Open an issue to discuss the design
  before a large pull request.

## How to contribute

1. Open an issue first for anything bigger than a small fix, so we can agree
   on the approach.
2. Keep pull requests focused on one change.
3. Run the tests for code changes: `pip install -e code/sim[test] -e code/training[test]`,
   `pytest code`, and `go test ./...` in `code/dsp`.
4. For text changes, render the chapter you changed with `quarto preview` and
   check that it builds.

## Writing style

- Write plainly, in short sentences, and explain terms on first use.
- Cite sources for claims about companies, products, and research, with a
  footnote that links to the source.
- Don't add anecdotes about real companies or people unless they are in the
  public record and cited.
- Say when a figure or number comes from the simulator or is illustrative.

## Licensing of contributions

The book's text and figures are licensed under CC BY-NC-SA 4.0
([LICENSE](LICENSE)), and the code under MIT ([LICENSE-CODE](LICENSE-CODE)).
By submitting a contribution, you agree that:

- your contribution is your own work, or you have the right to submit it;
- it is licensed under the same license as the part of the repository it
  changes (CC BY-NC-SA 4.0 for text and figures, MIT for code); and
- the author may also include it in any edition of the book, including
  commercial print editions, without further permission. You keep the
  copyright in your contribution.
