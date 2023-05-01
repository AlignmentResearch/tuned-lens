#################
Maintainers Guide
#################

++++++++++++++++++++++
The pre-merge workflow
++++++++++++++++++++++

The `pre-merge.yaml` workflow is designed to run basic linting on every PR. There are 4 major components to this workflow:

1. Ensuring that the pre-commit checks configured in :code:`.pre-commit-config.yaml` pass.
2. Ensuring that the package builds correctly and the :code:`pytest` tests pass on python versions 3.9 - 3.11.
3. Ensuring that this documentation builds correctly and the code within it runs including the tutorial notebooks.
4. Ensuring that the docker image builds correctly and uploading code coverage reports to `codecov <https://codecov.io/gh/AlignmentResearch/tuned-lens>`_.

Note: that the pre-merge workflow currently also runs on every push to the main branch. To make sure this passes is best practice to merge main into your branch before merging your PR.

+++++++++++++++++++
Publishing versions
+++++++++++++++++++

Publishing new versions is mostly handled by the CI here are the steps to follow to build and publish a new version:

1. To create a release first update the version in the :code:`pyproject.toml` then commit and push a tag of the form :code:`v<PEP440 Version>`. When making a new release it's a good idea to start with a pre-release version e.g. :code:`v0.0.5a0`.
* For more information on versioning see `PEP440 <https://www.python.org/dev/peps/pep-0440/>`_.
2. This will start the :code:`pre-release.yaml` workflow if it succeeds this will automatically create a `draft release <https://github.com/AlignmentResearch/tuned-lens/releases/>`_ in GitHub and publish the package to `test PyPI <https://test.pypi.org/project/tuned-lens/>`_.
* The pre-release
3. If you are happy with every thing, simply edit the newly created draft adding release notes etc and press the release button. This will run the `publish.yaml` workflow which publishes the package to `PyPI <https://pypi.org/project/tuned-lens/#description>`_ and uploads the docker image the GitHub package registry are synchronized.
4. Note: if this is **not** tagged as a pre-release version e.g. :code:`v0.0.5`, then pushing the tag should also automatically build the docs on `read the docs <https://readthedocs.org/projects/tuned-lens/versions/>`_.
