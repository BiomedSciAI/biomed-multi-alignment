import socket

import pytest

from mammal.examples.molnet.molnet_infer import load_model, task_infer


@pytest.mark.skipif(
    "ccc" not in socket.gethostname(),
    reason="Train consumes too much memory for a Travis run.",
)
def test_infer():
    smiles_seq = "C(Cl)Cl"
    for task_name in ["BBBP", "TOXICITY", "FDA_APPR"]:
        task_dict = load_model(task_name=task_name, device="cpu")
        result = task_infer(task_dict=task_dict, smiles_seq=smiles_seq)
        print(f"The prediction for {smiles_seq=} is {result}")
