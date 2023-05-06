#################
Maintainers Guide
#################

Here are some notes on how to maintain this package mostly focusing on the CI/CD workflow build on top of GitHub actions.

+++++++++++++++++++++++
The pull request checks
+++++++++++++++++++++++


The majority of the pull request checks are specified in the CI. Specifically, the `pre-merge.yaml <https://github.com/AlignmentResearch/tuned-lens/blob/improved-docs-85/.github/workflows/pre-merge.yml>`_ workflow. There are 4 major components to this workflow:

1. Ensuring that the pre-commit checks configured in ``.pre-commit-config.yaml`` pass.
2. Ensuring that the package builds correctly and the ``pytest`` tests pass on python versions 3.9 - 3.11.
    * ``pytest`` is configured in the ``pyproject.toml``.
3. Ensuring that this documentation builds correctly and the code within it runs including the tutorial notebooks.
    * The documentation is built using `sphinx <https://www.sphinx-doc.org/en/master/>`_ and the tutorial notebooks are run using `nbsphinx <https://nbsphinx.readthedocs.io/en/latest>`_.
4. Ensuring that the docker image builds correctly and uploading code coverage reports to `codecov <https://codecov.io/gh/AlignmentResearch/tuned-lens>`_.
    * The code coverage requirements themselves are contained in ``.codecov.yml``. Importantly, the code coverage bot itself enforce these requirements, not the CI.

Note that the pre-merge workflow also runs on every push to the main branch. To make sure this passes is best practice to merge main into the branch before merging your PR.

+++++++++++++++++++
Publishing versions
+++++++++++++++++++

Publishing new versions is mostly handled by the CI here are the steps to follow to build and publish a new version:

1. To create a release first update the version in the ``pyproject.toml`` then commit and push a tag of the form ``v<PEP440 Version>``. When making a new release it's a good idea to start with a pre-release version e.g. ``v0.0.5a0``.
    * For more information on versioning see `PEP440 <https://www.python.org/dev/peps/pep-0440/>`_.
2. This will start the `pre-release.yaml <https://github.com/AlignmentResearch/tuned-lens/blob/improved-docs-85/.github/workflows/pre-release.yml>`_ workflow if it succeeds this will automatically create a `draft release <https://github.com/AlignmentResearch/tuned-lens/releases/>`_ in GitHub and publish the package to `test PyPI <https://test.pypi.org/project/tuned-lens/>`_.
    * The specifically the pre-release workflow validates that the tag matches the version in the ``pyproject.toml``, and runs a very basic smoke test on the CI. It most of the heavy lifting is done by the `pre-merge.yaml <https://github.com/AlignmentResearch/tuned-lens/blob/improved-docs-85/.github/workflows/pre-merge.yml>`_ workflow.
3. If you are happy with every thing, simply edit the newly created draft adding release notes etc and press the release button. This will run the `publish.yaml <https://github.com/AlignmentResearch/tuned-lens/blob/improved-docs-85/.github/workflows/publish.yml>` workflow which publishes the package to `PyPI <https://pypi.org/project/tuned-lens/#description>`_ and uploads the docker image the GitHub package registry are synchronized.
4. Note that if ref is **not** tagged as a pre-release version e.g. ``v0.0.5``, then pushing the tag should also automatically build the docs on `read the docs <https://readthedocs.org/projects/tuned-lens/versions/>`_.
