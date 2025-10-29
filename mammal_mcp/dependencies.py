import logging
import os
from contextlib import asynccontextmanager

from fuse.data.tokenizers.modular_tokenizer.op import ModularTokenizerOp

# bmfm imports
from mammal.model import Mammal

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Create a logger
logger = logging.getLogger("mammal")
logger.setLevel(logging.DEBUG)

assets = {}


@asynccontextmanager
async def lifespan():
    if os.getenv("PROTEIN_PROTEIN_INTERACTION") == "true":
        # Load Model
        logger.info("downloading: ibm/biomed.omics.bl.sm.ma-ted-458m")
        model = Mammal.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m", cache_dir="model_cache"
        )
        assets["model"] = model.eval()
        logger.info("completed the download: ibm/biomed.omics.bl.sm.ma-ted-458m")

        logger.info("downloading for tokenizer: ibm/biomed.omics.bl.sm.ma-ted-458m")
        assets["tokenizer_op"] = ModularTokenizerOp.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m", cache_dir="model_cache"
        )

    if os.getenv("PROTEIN_SOLUBILITY") == "true":
        logger.info(
            "downloading: ibm-research/biomed.omics.bl.sm.ma-ted-458m.protein_solubility"
        )
        protein_solubility_model = Mammal.from_pretrained(
            "ibm-research/biomed.omics.bl.sm.ma-ted-458m.protein_solubility",
            cache_dir="model_cache",
        )
        assets["protein_solubility_model"] = protein_solubility_model.eval()
        assets["protein_solubility_tokenizer_op"] = ModularTokenizerOp.from_pretrained(
            "ibm-research/biomed.omics.bl.sm.ma-ted-458m.protein_solubility",
            cache_dir="model_cache",
        )

    if os.getenv("PROTEIN_DRUG_INTERACTION_MODEL") == "true":
        logger.info("downloading: ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd")
        protein_drug_interaction_model = Mammal.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd",
            cache_dir="model_cache",
        )
        assets["protein_drug_interaction_model"] = protein_drug_interaction_model.eval()
        assets["protein_drug_interaction_tokenizer_op"] = (
            ModularTokenizerOp.from_pretrained(
                "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd",
                cache_dir="model_cache",
            )
        )

    if os.getenv("TCR_EPITOPE_BINDING") == "true":
        logger.info(
            "downloading: ibm-research/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind"
        )

        tcr_epitope_model = Mammal.from_pretrained(
            "ibm-research/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind",
            cache_dir="model_cache",
        )
        #  set to eval/inference mode
        tcr_epitope_model.eval()
        #  set model to model mode (use 'cuda' if GPU avail)
        tcr_epitope_model.to(device="cpu")
        assets["tcr_epitope_model"] = tcr_epitope_model

        assets["tcr_epitope_model_tokenizer_op"] = ModularTokenizerOp.from_pretrained(
            "ibm-research/biomed.omics.bl.sm.ma-ted-458m.tcr_epitope_bind",
            cache_dir="model_cache",
        )

    if (
        os.getenv("DRUG_TARGET_BINDING") == "true"
        or os.getenv("DRUG_TARGET_BINDING_FASTA") == "true"
    ):
        logger.info("downloading: ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd")

        drug_target_model = Mammal.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd",
            cache_dir="model_cache",
        )
        #  set to eval/inference mode
        drug_target_model.eval()
        assets["drug_target_model"] = drug_target_model

        # download tokeniser
        assets["drug_target_model_tokeniser_op"] = ModularTokenizerOp.from_pretrained(
            "ibm/biomed.omics.bl.sm.ma-ted-458m.dti_bindingdb_pkd",
            cache_dir="model_cache",
        )

    logger.info("Assets loaded")

    yield
    # Clean up the assets
    assets.clear()
