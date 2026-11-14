"""Method -> Tool -> Education: teaching material compiled from the research model.

The educational outputs are generated from the same executable objects used for
the paper's analysis -- scijigsaw.cases.VAMP2, pcjigsaw.prob and the VAMP2 board
tables -- so a model revision regenerates the research result and the answer key
together. Every answer is computed; tests/test_teaching_pack.py pins them to the
manuscript.

    from pcjigsaw.education import build_vamp2_pack, build_pack
    build_vamp2_pack("vamp2_pack")          # four-step VAMP2 activity + advanced tasks
    build_pack("eif3", "eif3_pack")         # outreach + undergraduate pack for any small model

Command line: ``pcjigsaw-teach vamp2 OUTDIR`` and ``pcjigsaw-teach pack MODEL OUTDIR``.
"""
from .teach import TeachingModel, compute, build_pack, model_path, BUILTIN_MODELS
from .vamp2 import compute_vamp2, compute_gflownet, buildable, board, build as build_vamp2_pack

__all__ = ["TeachingModel", "compute", "build_pack", "model_path", "BUILTIN_MODELS",
           "compute_vamp2", "compute_gflownet", "buildable", "board", "build_vamp2_pack"]
