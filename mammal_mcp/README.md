# Mammal MCP server - making mammal tasks accessible to AI Agents.

A service that provides the `ibm/biomed.omics.bl.sm.ma-ted-458m` model tasks to AI Agents.

## Overview

MAMMAL (ibm/biomed.omics.bl.sm.ma-ted-458m) is a 'biomedical foundation model' (BMFM) that has been trained by IBM and the details can be found here -> https://github.com/BiomedSciAI/biomed-multi-alignment.

This repository is a fastmcp server which creates entrypoints for AI Agents to make inference for tasks currently supported by MAMMAL.

## Getting started

Change to mammal-mcp directory

```sh
cd mammal_mcp
```

Create the environment:

```sh
cp .env.example .env
```

These env vars control which modalities will be available to the agent.


The first time you run the server you need to download the models. Therefore set all the tasks that you will subsequently use to true in the .env and run:

```sh
uv run python -m server
```

Then wait for all the models to be downloaded before quiting the server.

## Running the server using STDIO (default)

### Integration into Claude Desktop

**If using Claude as your MCP client, DO NOT, add any confidential or personal data into the system.**

One of the easiest ways to experiment with the tools provided by mammal-mcp is to leverage the Claude Desktop.

For that, update your Claude Desktop config file (located at `~/Library/Application Support/Claude/claude_desktop_config.json`) with the JSON below:

```json
{
  "mcpServers": {
    "mammal": {
            "command": "<use output of `which uv` in the `mammal-mammal_mcp` folder>",
            "args": [
                "--directory",
                "<explicit path to mammal-mammal_mcp or use output `pwd` in `mammal-mammal_mcp` folder>",
                "run",
                "server.py"
            ]
        }
  }
}


```

**- Change both placeholders in this JSON indicated with <>**

Quit and re-start Claude Desktop, then use appropriate prompt.

### Integration into [MCPHost](https://github.com/mark3labs/mcphost) (using [Ollama](https://ollama.com/))

MCPHost is a host application that enables LLMs to interact with external tools through MCP.
It supports Claude 3.5 Sonnet and Ollama models, and we'll choose Ollama for this case as an example of using local LLM.

To install MCPHost, you'll need to install [go](https://go.dev/) as needed.
Then,

```sh
go install github.com/mark3labs/mcphost@latest
```

mcphost will be downloaded into the bin folder of go. You'll need to add PATH variable for that folder to launch it.

```sh
PATH="$(go env GOPATH)/bin:$PATH"
```

Install [Ollama](https://ollama.com/) on your desktop, and download the model you'd like to use. For example:

```sh
ollama run qwen3
```

Prepare configuration json file - make json file (for example mammal_mcp.json) as follows:

```json
{
  "mcpServers": {
    "mammal": {
      "command": "<use output of `which uv` in the `mammal-mammal_mcp` folder>",
      "args": [
        "--directory",
        "<explicit path to mammal-mammal_mcp or use output `pwd` in `mammal-mammal_mcp` folder>",
        "run",
        "server.py"
      ]
    }
  }
}
```

or, if you already have configuration json file for MCPHost, add "mammal" part above as the member of "mcpServers".

Finally, launch MCPHost using LLM which you'd like to use with the configuration file you prepared above. For example:

```sh
mcphost -m ollama:qwen3 --config <path to mammal_mcp.json you prepared above>
```

## Pre-trained task usage (in Claude)

Whichever task you want to utilize (we recommend not using anymore than 2 models at one time), first set this task to true in your .env file and then run:

```sh
uv run python -m server
```

This will pre-download the required models for your task. Once complete you will see the following message `Assets loaded`. You can kill the server at this point.

When you start Claude for any task you will get JSON parsing related error messages. This can be ignored.

### 1. Protein protein interaction prediction

- Binary classification task to predict protein-protein interaction using the pre-trained model `ibm-research/biomed.omics.bl.sm.ma-ted-458m`.

- Expected input are the either the amino acid sequences of the two proteins or the protein names.

- Ensure `PROTEIN_PROTEIN_INTERATION` is set to `true` in `.env` file

Example prompt:

```
Do proteins VPS35 and VPS26 interact together?
```

### 2. Protein solubility prediction

- Binary classification task to predict protein solubility using the fine-tuned model `ibm-research/biomed.omics.bl.sm.ma-ted-458m.protein_solubility`.

- Expected input are the either the amino acid sequences of the protein or the protein name.

- Ensure `PROTEIN_SOLUBILITY` is set to `true` in `.env` file

Example prompt:

```
Is protein VPS35 soluble in aqueous solutions?
```
### 3. Drug-target binding prediction

- Prediction of drug-target binding affinity using the fine-tuned model `ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd`.

- Expected input are the amino acid sequence of the target and the SMILES representation of the drug. Binding affinity is predicted using pKd (the negative logarithm of the dissociation constant, reflecting the strength of the interaction between a sm molecule and protein)

- Ensure `DRUG_TARGET_BINDING` is set to `true` in `.env` file

Example prompt:

```
What is the predicted binding affinity between the drug with the SMILES sequence "CC(=O)NCCC1=CNc2c1cc(OC)cc2" and target protein with amino acid sequence "NLMKRCTRGFRKLGKCTTLEEEKCKTLYPRGQCTCSDSKMNTHSCDCKSC"
```

### 4. TCR-epitope binding

- Binary classification task predicting binding of binding between T-cell receptor and epitope sequences using the fine-tuned model `ibm-research/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind`.

- Expected inputs are the amino acid sequences of the epitope and T-cell receptor.

- Ensure `TCR_EPITOPE_BINDING` is set to `true` in `.env` file

Example prompt:

```
does the tcr with the following sequence NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSWDRVLEQYFGPGTRLTVT bind to the epitope with following sequence LLQTGIHVRVSQPSL?
```

## Running the server using Streamable-HTTP

Change the STREAMABLE_HTTP environment variable in the .env file to true.

Then run:

```sh
uv run python -m server
```

The server should start on http://127.0.0.1:8001 (if you want to change the port number from 8001 then modify this in the .env file) after loading all the models for the tasks you have selected.
