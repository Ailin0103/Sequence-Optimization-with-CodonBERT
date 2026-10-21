# CodonBERT Streamlit demo

Public-facing UI for the released Cambray protein-expression regression model.
The app accepts DNA/RNA coding sequences, validates codon alignment and length,
and reverses target standardization to the original `clean.lin.prot.mean` scale.
No training data, test-set evaluation, or credentials are needed for inference.

## Run locally

From the repository root, create a separate Python 3.11 environment:

```bash
python3.11 -m venv /tmp/codonbert-streamlit
/tmp/codonbert-streamlit/bin/python -m pip install -r demo/requirements.txt
/tmp/codonbert-streamlit/bin/python -m streamlit run demo/app.py
```

The model is downloaded on first prediction and cached in memory. Linux cloud
installations use the CPU PyTorch wheel index to avoid unnecessary CUDA libraries.

## Free public deployment: Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ with your GitHub account.
2. Click Create app and select:
   - Repository: `Ailin0103/Sequence-Optimization-with-CodonBERT`
   - Branch: `main`
   - Main file path: `demo/app.py`
3. In Advanced settings select Python **3.11**. No secrets are required.
4. Choose an available app subdomain and deploy.
5. Verify `ATGGCC` returns approximately **36.9457**. Invalid `ATN` should
   display a validation error. This is a smoke test, not biological validation.

Streamlit finds `demo/requirements.txt` next to the entry point; do not use the
research project's full Pixi environment for cloud deployment. Free hosting has
resource and sleep limits. Successful local testing does not establish that the
free cloud memory limit is sufficient; check the deployment logs and real inference.
The cloud app is not public until account sign-in and deployment are completed.

## Model and limitations

Model: https://huggingface.co/Ailin0103/CodonBERT-Cambray-expression-regression

The code pins the verified weight revision. Training sequences were 96 nt;
external generalization is unestablished. The 6 nt example verifies execution
only. Academic/non-commercial upstream restrictions apply. No sequences are
intentionally stored by the app, but hosting-provider request logs may exist.
