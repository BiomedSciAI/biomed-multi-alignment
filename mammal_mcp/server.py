import asyncio
import logging
import os

import requests
import torch
from Bio import SeqIO
from dependencies import assets, lifespan
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel
from util import process_model_output

from mammal.examples.dti_bindingdb_kd.task import DtiBindingdbKdTask
from mammal.examples.protein_solubility.task import ProteinSolubilityTask
from mammal.examples.tcr_epitope_binding.main_infer import task_infer
from mammal.keys import (
    CLS_PRED,
    ENCODER_INPUTS_ATTENTION_MASK,
    ENCODER_INPUTS_STR,
    ENCODER_INPUTS_TOKENS,
    SCORES,
)

load_dotenv()

# Create an MCP server
mcp = FastMCP("mammal")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Create a logger
logger = logging.getLogger("mammal")
logger.setLevel(logging.DEBUG)


def fetch_protein_sequence(gene_name: str):
    """
    This function will search for a single protein amino acid sequence.
    :param protein_name: the name of the protein all uppercase
    :return: string value representing the amino acid sequence
    """
    url = f"https://rest.uniprot.org/uniprotkb/{gene_name}_HUMAN"
    params = {"fields": "sequence"}
    headers = {"accept": "application/json"}

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data_object = response.json()

        sequence = data_object.get("sequence", "Sequence not found")
        return sequence["value"]
    except Exception as e:
        print(f"Error fetching data: {e}")
        raise


class GeneSequenceInformation(BaseModel):
    gene_name: str
    protein_amino_acid_sequence: str


@mcp.tool()
async def gene_name_to_amino_acid_sequence(gene_name: str) -> GeneSequenceInformation:
    """
    Retrieves amino acid sequences for one gene asynchronously.

    This function takes one required gene name , and returns
    a GeneSequenceInformation object containing the amino acid sequences for
    the specified genes. If the second gene name is empty, None, or "null", only the
    first gene will be processed.

    Args:
        gene_name (str): The name of the gene to retrieve sequence information for.

    Returns:
        GeneSequenceInformation: A GeneSequenceInformation object containing
        amino acid sequences and related information for the requested gene(s).

    Example:
        result = await gene_name_to_amino_acid_sequence("BRCA1")
    """

    logger.info(f"gene_name: {gene_name}")
    return GeneSequenceInformation(
        gene_name=gene_name,
        protein_amino_acid_sequence=fetch_protein_sequence(gene_name),
    )


def single_protein_protein_interaction_prediction(
    protein_aa_sequence_1: str, protein_aa_sequence_2: str, assets: dict
) -> float:
    # Create and load sample
    sample_dict = dict()
    # Formatting prompt to match pre-training syntax
    sample_dict[ENCODER_INPUTS_STR] = (
        f"<@TOKENIZER-TYPE=AA><BINDING_AFFINITY_CLASS><SENTINEL_ID_0><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{protein_aa_sequence_1}<SEQUENCE_NATURAL_END><MOLECULAR_ENTITY><MOLECULAR_ENTITY_GENERAL_PROTEIN><SEQUENCE_NATURAL_START>{protein_aa_sequence_2}<SEQUENCE_NATURAL_END><EOS>"
    )

    # Tokenize
    tokenizer_op = assets["tokenizer_op"]
    tokenizer_op(
        sample_dict=sample_dict,
        key_in=ENCODER_INPUTS_STR,
        key_out_tokens_ids=ENCODER_INPUTS_TOKENS,
        key_out_attention_mask=ENCODER_INPUTS_ATTENTION_MASK,
    )
    sample_dict[ENCODER_INPUTS_TOKENS] = torch.tensor(
        sample_dict[ENCODER_INPUTS_TOKENS]
    )
    sample_dict[ENCODER_INPUTS_ATTENTION_MASK] = torch.tensor(
        sample_dict[ENCODER_INPUTS_ATTENTION_MASK]
    )

    # Generate Prediction
    evaluation_model = assets["model"]
    batch_dict = evaluation_model.generate(
        [sample_dict],
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=5,
    )

    ans = process_model_output(
        tokenizer_op=tokenizer_op,
        decoder_output=batch_dict[CLS_PRED][0],
        decoder_output_scores=batch_dict[SCORES][0],
    )

    return ans["normalized_scores"].item()


def drug_target_pkd_prediction_single(
    target_seq: str, drug_seq: str
) -> dict[str, float]:

    modelpath = assets["drug_target_model"]  # load pre-loaded model
    tokenizerpath = assets["drug_target_model_tokeniser_op"]

    # convert to MAMMAL style
    sample_dict = {"target_seq": target_seq, "drug_seq": drug_seq}
    sample_dict = DtiBindingdbKdTask.data_preprocessing(
        sample_dict=sample_dict,
        tokenizer_op=tokenizerpath,
        target_sequence_key="target_seq",
        drug_sequence_key="drug_seq",
        norm_y_mean=None,
        norm_y_std=None,
        device=modelpath.device,
    )

    # forward pass - encoder only mode which supports scalar predictions
    batch_dict = modelpath.forward_encoder_only([sample_dict])

    # post-process model output
    batch_dict = DtiBindingdbKdTask.process_model_output(
        batch_dict,
        scalars_preds_processed_key="model.out.dti_bindingdb_kd",
        norm_y_mean=5.79384684128215,
        norm_y_std=1.33808027428196,
    )

    result = {
        "model.out.dti_bindingdb_kd": float(batch_dict["model.out.dti_bindingdb_kd"][0])
    }
    return result


def drug_target_pkd_prediction_fasta(
    target_file: str, drug_sequence: str
) -> dict[str, float]:

    result = {}
    for seq_record in SeqIO.parse(target_file, "fasta"):
        if len(seq_record) > 200 and len(seq_record) < 1240:
            sequence_name = seq_record.description
            # sample_dict = {"target_seq": target_seq, "drug_seq": drug_seq}
            batch_dict = drug_target_pkd_prediction_single(
                seq_record.seq, drug_sequence
            )
            result[sequence_name] = batch_dict["model.out.dti_bindingdb_kd"]
        # result = {"model.out.dti_bindingdb_kd" : float(batch_dict["model.out.dti_bindingdb_kd"][0])}
    return result


if os.getenv("PROTEIN_PROTEIN_INTERACTION") == "true":

    @mcp.tool()
    async def protein_protein_interaction(
        protein_aa_sequence_1: str, protein_aa_sequence_2: str
    ) -> float:
        """
        This function can be used to predict the interaction probability between two protein amino acid sequences.
        :param protein_aa_sequence_1: str
        :param protein_aa_sequence_2: str
        :return: An interaction score. 1 is that the proteins do interact, 0 is that the proteins do NOT interact.
        """

        return single_protein_protein_interaction_prediction(
            protein_aa_sequence_1, protein_aa_sequence_2, assets
        )


if os.getenv("PROTEIN_SOLUBILITY") == "true":

    @mcp.tool()
    async def protein_solubility(protein_aa_sequence: str) -> float:
        """
        This function can be used to predict protein solubility.
        :param protein_amino_acid_sequence: str
        :return: solubility factor: float
        """
        sample_dict = {"protein_seq": protein_aa_sequence}
        tokenizer_op = assets["protein_solubility_tokenizer_op"]

        sample_dict = ProteinSolubilityTask.data_preprocessing(
            sample_dict=sample_dict,
            protein_sequence_key="protein_seq",
            tokenizer_op=tokenizer_op,
        )

        evaluation_model = assets["protein_solubility_model"]

        batch_dict = evaluation_model.generate(
            [sample_dict],
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=5,
        )

        ans = ProteinSolubilityTask.process_model_output(
            tokenizer_op=tokenizer_op,
            decoder_output=batch_dict[CLS_PRED][0],
            decoder_output_scores=batch_dict[SCORES][0],
        )

        # Print prediction
        return ans["normalized_scores"].item()


if os.getenv("TCR_EPITOPE_BINDING") == "true":

    @mcp.tool()
    async def tcr_epitope_binding(
        tcr_beta_seq: str, epitope_seq: str
    ) -> dict[str, float]:
        """
        Add docstring
        """
        modelpath = assets["tcr_epitope_model"]  # load pre-loaded model
        tokenizerpath = assets[
            "tcr_epitope_model_tokenizer_op"
        ]  # load tokeniser operater

        result = task_infer(
            model=modelpath,
            tokenizer_op=tokenizerpath,
            tcr_beta_seq=tcr_beta_seq,
            epitope_seq=epitope_seq,
        )
        return result


if os.getenv("DRUG_TARGET_BINDING") == "true":

    @mcp.tool()
    async def drug_target_binding(target_seq: str, drug_seq: str) -> dict[str, float]:
        """
        This tool predicts drug-target binding affinity using the fine-tuned model `ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd`.

        Expected input are the amino acid sequence of the target and the SMILES representation of the drug.

        inding affinity is predicted using pKd (the negative logarithm of the dissociation constant, reflecting the strength of the interaction between a sm molecule and protein)

        """
        return drug_target_pkd_prediction_single(target_seq, drug_seq)


if os.getenv("DRUG_TARGET_BINDING_FASTA") == "true":

    @mcp.tool()
    async def drug_target_binding_fasta(
        target_file: str, drug_sequence: str
    ) -> dict[str, float]:
        """
        This tool predicts drug-target binding affinity using the fine-tuned model `ibm-research/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd`.

        Expected input is a fasta file of multiple amino acid sequences targets and a single SMILES representation of the drug.

        Binding affinity for each sequence is predicted using pKd (the negative logarithm of the dissociation constant, reflecting the strength of the interaction between a sm molecule and protein)

        """
        return drug_target_pkd_prediction_fasta(target_file, drug_sequence)


async def main():
    async with lifespan():
        # Initialize and run the server once assets are loaded
        if os.getenv("STREAMABLE_HTTP") == "true":
            logger.info("starting up mammal MCP server with streamable-http...")
            await mcp.run_async(
                transport="streamable-http",
                host="127.0.0.1",
                port=int(os.getenv("PORT")),
            )
        if os.getenv("SSE") == "true":
            logger.info("starting up mammal MCP server with sse...")
            await mcp.run_async(
                transport="sse",
                host="127.0.0.1",
                port=int(os.getenv("PORT")),
            )
        else:
            logger.info("starting up mammal MCP-server stdio...")
            await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
