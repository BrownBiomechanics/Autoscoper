Contributing to Autoscoper
==========================

There are many ways to contribute to Autoscoper, with varying levels of effort. Do try to
look through the [documentation](https://github.com/BrownBiomechanics/Autoscoper#readme) first if something is unclear, and let us know how we can
do better.

  * Ask a question on the [Autoscoper forum][autoscoper-forum]
  * Use [Autoscoper issues][autoscoper-issues] to submit a feature request or bug, or add to the discussion on an existing issue
  * Submit a [Pull Request](https://github.com/BrownBiomechanics/Autoscoper/pulls) to improve Autoscoper or its documentation

We encourage a range of Pull Requests, from patches that include passing tests and
documentation, all the way down to half-baked ideas that launch discussions.

The PR Process, Circle CI, and Related Gotchas
----------------------------------------------

#### How to submit a PR ?

If you are new to Autoscoper development and you don't have push access to the Autoscoper
repository, here are the steps:

1. [Fork and clone](https://docs.github.com/get-started/quickstart/fork-a-repo) the repository.
2. Create a branch.
3. [Push](https://docs.github.com/get-started/using-git/pushing-commits-to-a-remote-repository) the branch to your GitHub fork.
4. Create a [Pull Request](https://github.com/BrownBiomechanics/Autoscoper/pulls).

This corresponds to the `Fork & Pull Model` described in the [GitHub collaborative development](https://docs.github.com/pull-requests/collaborating-with-pull-requests/getting-started/about-collaborative-development-models)
documentation.

When submitting a PR, the developers following the project will be notified. That
said, to engage specific developers, you can add `Cc: @<username>` comment to notify
them of your awesome contributions.
Based on the comments posted by the reviewers, you may have to revisit your patches.


#### How to efficiently contribute ?

We encourage all developers to:

* add or update tests.

* consider potential backward compatibility breakage and discuss these on the
  [Autoscoper forum][autoscoper-forum].

#### How to write commit messages ?

Write your commit messages using the standard prefixes for Autoscoper commit
messages:

  * `BUG:` Fix for runtime crash or incorrect result
  * `COMP:` Compiler error or warning fix
  * `DOC:` Documentation change
  * `ENH:` New functionality
  * `PERF:` Performance improvement
  * `STYLE:` No logic impact (indentation, comments)
  * `WIP:` Work In Progress not ready for merge

The body of the message should clearly describe the motivation of the commit
(**what**, **why**, and **how**). In order to ease the task of reviewing
commits, the message body should follow the following guidelines:

  1. Leave a blank line between the subject and the body.
  This helps `git log` and `git rebase` work nicely, and allows to smooth
  generation of release notes.
  2. Try to keep the subject line below 72 characters, ideally 50.
  3. Capitalize the subject line.
  4. Do not end the subject line with a period.
  5. Use the imperative mood in the subject line (e.g. `BUG: Fix spacing
  not being considered.`).
  6. Wrap the body at 80 characters.
  7. Use semantic line feeds to separate different ideas, which improves the
  readability.
  8. Be concise, but honor the change: if significant alternative solutions
  were available, explain why they were discarded.
  9. If the commit refers to a topic discussed on the [Autoscoper forum][autoscoper-forum], or fixes
  a regression test, provide the link. If it fixes a compiler error, provide a
  minimal verbatim message of the compiler error. If the commit closes an
  issue, use the [GitHub issue closing
  keywords](https://docs.github.com/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue).

Keep in mind that the significant time is invested in reviewing commits and
*pull requests*, so following these guidelines will greatly help the people
doing reviews.

These guidelines are largely inspired by Chris Beam's
[How to Write a Commit Message](https://chris.beams.io/posts/git-commit/)
post.

Examples:
  - Bad: `BUG: Check pointer validity before dereferencing` -> implementation detail, self-explanatory (by looking at the code)
  - Good: `BUG: Fix crash in Module X when clicking Apply button`
  - Bad: `ENH: More work in CameraViewWidget` -> more work is too vague, CameraViewWidget is too low level
  - Good: `ENH: Add float image outputs in module X`
  - Bad: `COMP: Typo in cmake variable` -> implementation detail, self-explanatory
  - Good: `COMP: Fix compilation error with Numpy on Visual Studio`


#### How to integrate a PR ?

Getting your contributions integrated is relatively straightforward, here
is the checklist:

* All tests pass
* Consensus is reached. This usually means that at least two reviewers approved
  the changes (or added a `LGTM` comment) and at least one business day passed
  without anyone objecting. `LGTM` is an acronym for _Looks Good to Me_.
* To accommodate developers explicitly asking for more time to test the
  proposed changes, integration time can be delayed by few more days.

* If you do NOT have push access, a Autoscoper core developer will integrate your PR. If
  you would like to speed up the integration, do not hesitate to send a note on
  the [Autoscoper forum][autoscoper-forum].

#### Decision-making process

1. Given the topic of interest, initiate discussion on the [Autoscoper forum][autoscoper-forum].

2. Identify a small circle of community members that are interested to study the
   topic in more depth.

3. Take the discussion off the general list, work on the analysis of options and
   alternatives, summarize findings on the wiki or similar.

4. Announce on the [Autoscoper forum][autoscoper-forum] the in-depth discussion of the topic for the
   Autoscoper bi-weekly meeting,
   encourage anyone that is interested in weighing in on the topic to join the
   discussion. If there is someone who is interested to participate in the discussion,
   but cannot join the meeting due to conflict, they should notify the leaders of
   the given project and identify the time suitable for everyone.

5. Hopefully, reach consensus at the hangout and proceed with the agreed plan.


*The initial version of these guidelines is adapted from [3D Slicer guidlines](https://slicer.readthedocs.io/en/latest/developer_guide/contributing.html?highlight=contributing#decision-making-process)*


[slicer-forum]: https://discourse.slicer.org/c/community/slicerautoscoperm/30
[slicer-issues]: https://github.com/BrownBiomechanics/Autoscoper/issues
