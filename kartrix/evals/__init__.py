"""The agent and RAG eval suite (step 2.3): measures Kartrix itself — not the apps users build.

``kartrix eval rag|agent|all|calibrate|report`` (see :mod:`kartrix.evals.cli`). Datasets live in the
repository's ``evals/`` folder, so the evals run from a Kartrix source checkout. DeepEval provides
the LLM-as-judge metrics (with Kartrix's own judge model); retrieval and trajectory metrics are
computed here.
"""
