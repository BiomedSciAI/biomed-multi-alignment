from mammal.examples.TCR_epitope_binding.main_infer import load_model, task_infer


def test_infer() -> None:
    """
    A test for TCR beta chain and epitope binding example on HF, https://huggingface.co/ibm/biomed.omics.bl.sm.ma-ted-458m
    """
    # positive 1:
    TCR_beta_seq = "NAGVTQTPKFQVLKTGQSMTLQCAQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSWDRVLEQYFGPGTRLTVT"
    epitope_seq = "LLQTGIHVRVSQPSL"

    # positive 2:
    # TCR_beta_seq = "GAVVSQHPSWVICKSGTSVKIECRSLDFQATTMFWYRQFPKQSLMLMATSNEGSKATYEQGVEKDKFLINHASLTLSTLTVTSAHPEDSSFYICSASEGTSSYEQYFGPGTRLTVT"
    # epitope_seq = "FLKEKGGL"

    task_name = "TCR_epitope_bind"
    task_dict = load_model(task_name=task_name, device="cpu")
    result = task_infer(
        task_dict=task_dict, TCR_beta_seq=TCR_beta_seq, epitope_seq=epitope_seq
    )
    print(f"The prediction for {epitope_seq} and {TCR_beta_seq} is {result}")


if __name__ == "__main__":
    test_infer()
