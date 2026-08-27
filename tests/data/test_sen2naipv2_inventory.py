from pathlib import Path

from trustsr.data.provenance import load_dataset_source


def test_pinned_sen2naipv2_inventory_matches_spec() -> None:
    artifact = Path(__file__).parents[2] / "artifacts/datasets/sen2naipv2-source-v1.json"
    source = load_dataset_source(artifact)

    assert source.schema == "trustsr.sen2naipv2-source.v1"
    assert source.repository == "tacofoundation/SEN2NAIPv2"
    assert source.revision == "c370504201072fdb1dd388013ab8c0fc7d00a57e"
    assert source.license_claim == "cc0-1.0"
    assert source.card_sha256 == (
        "5897aed9410fef305953ff5b34e83697b466901583b880158af2902a8267a58d"
    )
    assert source.bands == ("B04", "B03", "B02", "B08")
    assert source.scale == 4
    assert source.lr_shape == (130, 130)
    assert source.hr_shape == (520, 520)

    expected_objects = (
        (
            "sen2naipv2-crosssensor.taco",
            "c6f29d8e80dc5e856e2b4510c0e6830043d4b15c9228a9ca249a4f618e7475a5",
            9_717_583_850,
        ),
        (
            "sen2naipv2-histmatch.0000.part.taco",
            "a276024df0f81ff53770cf1b415d0f86268bd2b090a467b80e2e8b3992d08acc",
            20_000_560_299,
        ),
        (
            "sen2naipv2-histmatch.0001.part.taco",
            "c493107a10a488643346aba990717e2caa839f26a3cbde4e359c7f0f83158c4b",
            19_999_654_192,
        ),
        (
            "sen2naipv2-histmatch.0002.part.taco",
            "f922d29d7701cbcad6e41da235805c26ad11200f50acc34e77af7de38abd66ae",
            19_999_987_735,
        ),
        (
            "sen2naipv2-histmatch.0003.part.taco",
            "3faadd2e3b9b9a9611764e9e4f8c2d230667d6d75604eb26d82e5e8f1e65da26",
            9_361_318_942,
        ),
        (
            "sen2naipv2-unet.0000.part.taco",
            "b8f7a8497328e62fcb53872d2d59220a0fc84fd05f6d205bbe54b4c2d32fa6c2",
            20_000_562_604,
        ),
        (
            "sen2naipv2-unet.0001.part.taco",
            "c24c4500f779f0d9db6d7c2f568879dd3c40d26fd8e36cde1b235dfecb489e1f",
            20_000_506_401,
        ),
        (
            "sen2naipv2-unet.0002.part.taco",
            "aba2f08677464e8359e5c98d7ebe77bfab623919a7cde3c24f8e0018dc4319ce",
            20_000_130_510,
        ),
        (
            "sen2naipv2-unet.0003.part.taco",
            "8835cbf9d0d179190f495802e2b548e4cf1d45b7c18d8910b31a2449c9b49632",
            10_275_824_059,
        ),
    )

    assert len(source.objects) == 9
    assert tuple((obj.path, obj.sha256, obj.size_bytes) for obj in source.objects) == (
        expected_objects
    )
    assert source.total_bytes == 149_356_128_592
    assert source.declared_total_bytes == source.total_bytes
