---
name: 🛠️ Feature Request
description: Suggest an idea to help us improve MAMMAL
title: "[Feature]: "
labels:
  - "ty:feature"

body:
  - type: markdown
    attributes:
      value: >
        **Thanks :heart: for taking the time to fill out this feature request report!** We kindly ask that you search to
        see if an issue [already exists](https://github.com/BiomedSciAI/biomed-multi-alignment/issues) for
        your feature.

        We are also happy to accept contributions from our users. For more details see
        [here](https://github.com/BiomedSciAI/biomed-multi-alignment/blob/main/CONTRIBUTING.md).

  - type: textarea
    attributes:
      label: Description
      description: |
        A clear and concise description of the feature you're interested in.
      value: |
        <!--- Describe your feature here --->
    validations:
      required: true

  - type: textarea
    attributes:
      label: Suggested Solution
      description: >
        Describe the solution you'd like. A clear and concise description of what you want to happen. If you have
        considered alternatives, please describe them.
      value: |
        <!--- Describe your solution here --->
    validations:
      required: false
