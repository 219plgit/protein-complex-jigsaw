def test_pcjigsaw_namespace_imports():
    import pcjigsaw
    assert pcjigsaw.__version__ == "4.0.0"
    assert hasattr(pcjigsaw, "Assembly")
    assert hasattr(pcjigsaw, "analyse_perturbation")


def test_failure_does_not_imply_aggregation():
    from pcjigsaw import Assembly, Perturbation, analyse_perturbation, assess_misassembly
    a = Assembly(requires={"B": {"A"}, "A": set()})
    report = analyse_perturbation(a, Perturbation("A unavailable", unavailable_components=("A",)))
    m = assess_misassembly(report)
    assert m.aggregation_inferred is False
    assert m.supported_off_pathway_outcomes == ()
