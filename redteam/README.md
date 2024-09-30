# MLCommons modeltune red team tooling

A set of generative AI red team tooling to assist in MLCommons safety model development

## Components
1. Discovery: Finding vulnerabilities for a particular model
1. Inventory: Classify and track a database of jaibreak methods per model that we use
1. Application: Applying jailbreak methods for MLC safety model development

## How red teaming helps safety model development
- Generating dangerous SUT responses by creating variants of the human written prompts that we recieve
- Synthetic dataset generation to augment existing datasets with difficult samples for target SUTs

