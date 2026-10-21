"""Public, single-sequence demonstration of the released regression model."""
import streamlit as st
import json
import math
import re
from pathlib import Path

MODEL_ID = 'Ailin0103/CodonBERT-Cambray-expression-regression'
REVISION = '2ce33e5883615b82bd5d724642ab4ace4512a001'


def format_sequence(sequence):
    rna = re.sub(r'\s+', '', sequence or '').upper().replace('T', 'U')
    if not rna:
        raise ValueError('Paste a DNA or RNA coding sequence first.')
    if set(rna) - set('AUGC'):
        raise ValueError('Use only A, C, G, T or U. Remove FASTA headers and ambiguous bases.')
    if len(rna) % 3:
        raise ValueError('Sequence length must be a multiple of 3 nucleotides.')
    if len(rna) > 186:
        raise ValueError('Maximum length is 186 nucleotides (62 codons). No truncation is applied.')
    return ' '.join(rna[i:i + 3] for i in range(0, len(rna), 3)), len(rna)


@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer, BertForSequenceClassification
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = BertForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION).eval()
    with open(hf_hub_download(MODEL_ID, 'target_scaler.json', revision=REVISION)) as handle:
        scaler = json.load(handle)
    return tokenizer, model, scaler


def predict(sequence):
    import torch
    formatted, length = format_sequence(sequence)
    tokenizer, model, scaler = load_model()
    inputs = tokenizer(formatted, return_tensors='pt', truncation=False)
    with torch.inference_mode():
        prediction_z = model(**inputs).logits.item()
    prediction = prediction_z * scaler['target_std'] + scaler['target_mean']
    if not math.isfinite(prediction):
        raise RuntimeError('The model returned an invalid prediction. Please try again.')
    return round(prediction, 4), f'{length} nucleotides · {length // 3} codons. Output is on the original clean.lin.prot.mean scale.'


def load_reference():
    return json.loads(Path(__file__).with_name('expression_reference.json').read_text())


def expression_score(expression, reference):
    raw_score = 100 * (expression - reference['minimum']) / (reference['maximum'] - reference['minimum'])
    return max(0.0, min(100.0, raw_score)), raw_score


def main():
    st.set_page_config(page_title='CodonBERT · Protein expression', page_icon='🧬', layout='centered')
    reference = load_reference()
    st.caption('CODONBERT · RESEARCH DEMO')
    st.title('Predict protein expression')
    st.write('Paste a DNA or RNA coding sequence. This model was fine-tuned on the Cambray synthetic **E. coli** library.')
    st.markdown('[Model & inference guide](https://huggingface.co/Ailin0103/CodonBERT-Cambray-expression-regression) · '
                '[Project](https://github.com/Ailin0103/Sequence-Optimization-with-CodonBERT)')
    with st.form('expression_form'):
        sequence = st.text_area('Coding sequence (DNA or RNA)', height=160,
                                placeholder='Paste A/C/G/T or A/C/G/U sequence here…',
                                help='Whitespace is ignored. No FASTA headers. Length must be divisible by 3 and no more than 186 nt.')
        submitted = st.form_submit_button('Predict expression', type='primary')
    st.caption('Technical example: ATGGCC. This short example checks the interface; it is not a validated biological design.')
    if submitted:
        try:
            # Validate before downloading or loading the model.
            format_sequence(sequence)
            with st.spinner('Loading model and predicting… First use may take longer.'):
                expression, detail = predict(sequence)
            score, raw_score = expression_score(expression, reference)
            st.metric('Relative expression score (0–100)', f'{score:.2f} / 100')
            st.progress(score / 100)
            st.caption(f'Original-scale prediction: {expression:.4f} · clean.lin.prot.mean')
            if raw_score < 0 or raw_score > 100:
                st.info('The original prediction falls outside the observed training range. '
                        'The displayed score is capped at 0 or 100; the original prediction is shown unchanged.')
            st.caption(detail)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('Prediction could not complete. The server may be starting or have insufficient memory. Please retry later.')
    st.subheader('Score reference')
    st.markdown(f"**0 points** = training minimum ({reference['minimum']:.2f}) · "
                f"**50 points** = range midpoint ({(reference['minimum'] + reference['maximum']) / 2:.2f}) · "
                f"**100 points** = training maximum ({reference['maximum']:.2f})")
    median_score, _ = expression_score(reference['median'], reference)
    p05_score, _ = expression_score(reference['p05'], reference)
    p95_score, _ = expression_score(reference['p95'], reference)
    st.caption(f"Based on {reference['count']:,} training sequences only. Training median: {median_score:.2f} points; "
               f"middle 90% of observed training values: {p05_score:.2f}–{p95_score:.2f} points.")
    with st.expander('How to interpret the result', expanded=True):
        st.write('Higher scores mean higher predicted expression, not better fitness, confidence, accuracy, or a percentile. '
                 'The 0–100 display is a linear rescaling of the training range, not the model’s training z-score.')
        st.code('score = 100 × (original_prediction − 1) / (100 − 1)', language=None)
        st.write('The model output is first converted back to clean.lin.prot.mean using training mean and standard deviation. '
                 'The display score is then capped to 0–100. Original predictions remain visible. '
                 'The reference range describes training data; it is not a prediction interval or physical concentration.')
        st.write('Training sequences were 96 nt. Other supported lengths may be outside the training distribution. '
                 'Generalization beyond this synthetic E. coli library has not been established.')
        st.caption('Published held-out test RMSE: 14.0871. This is a dataset-level metric, not an uncertainty interval for your sequence.')
    st.divider()
    st.caption('Research demonstration · Academic/non-commercial restrictions apply. '
               'Inputs are sent to the hosting server. This app does not intentionally save sequences; '
               'hosting-provider infrastructure may retain request logs.')


if __name__ == '__main__':
    main()
