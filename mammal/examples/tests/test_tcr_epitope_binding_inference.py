from mammal.examples.tcr_epitope_binding.main_infer import load_model, task_infer


def test_infer() -> None:
    """
    A test for TCR beta chain and epitope binding example on HF, https://huggingface.co/ibm/biomed.omics.bl.sm.ma-ted-458m
    """
    # positive 1:
    tcr_beta_seq = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSWDRVLEQYFGPGTRLTVT"
    epitope_seq = "LLQTGIHVRVSQPSL"

    # positive 2:
    # tcr_beta_seq = "GAVVSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSASEGTSSYEQYFGPGTRLTVT"
    # epitope_seq = "FLKEKGGL"

    model_inst, tokenizer_op = load_model(device="cpu")
    result = task_infer(
        model=model_inst,
        tokenizer_op=tokenizer_op,
        tcr_beta_seq=tcr_beta_seq,
        epitope_seq=epitope_seq,
    )
    print(f"The prediction for {epitope_seq} and {tcr_beta_seq} is {result}")


if __name__ == "__main__":
    test_infer()
