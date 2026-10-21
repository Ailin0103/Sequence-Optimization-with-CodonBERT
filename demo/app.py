"""Public, single-sequence demonstration of the released regression model."""
import streamlit as st
import json
import math
import re

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


def render_score_bar(expression=None):
    position = 0.0 if expression is None else (max(1.0, min(100.0, expression)) - 1.0) / 99.0 * 100
    st.markdown(f"""
<div style="margin:16px 0 24px">
  <div style="font-weight:600;margin-bottom:12px">Expression score · 1–100</div>
  <div role="img" aria-label="Expression scale from 1 to 100" style="height:18px;border-radius:9px;background:#e5e7eb;overflow:hidden">
    <div style="height:100%;width:{position:.4f}%;background:#14b8a6;border-radius:9px"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:8px;font-size:18px;font-weight:700">
    <span>1</span><span>100</span>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:13px;opacity:0.7">
    <span>Lower expression</span><span>Higher expression</span>
  </div>
</div>
""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='CodonBERT · Protein expression', page_icon='🧬', layout='centered')
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
        st.session_state.pop('expression_prediction', None)
        try:
            # Validate before downloading or loading the model.
            format_sequence(sequence)
            with st.spinner('Loading model and predicting… First use may take longer.'):
                expression, _ = predict(sequence)
            st.session_state['expression_prediction'] = expression
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error('Prediction could not complete. The server may be starting or have insufficient memory. Please retry later.')
    expression = st.session_state.get('expression_prediction')
    if expression is not None:
        st.metric('Predicted expression', f'{expression:.2f} / 100')
    render_score_bar(expression)
    if expression is not None and (expression < 1 or expression > 100):
        st.caption('Prediction is outside the training range; the bar stops at the nearest endpoint.')
    st.caption('Higher values mean higher predicted expression. Research estimate for related synthetic E. coli sequences.')
    st.divider()
    st.caption('Research demonstration · Academic/non-commercial restrictions apply. '
               'Inputs are sent to the hosting server. This app does not intentionally save sequences; '
               'hosting-provider infrastructure may retain request logs.')


if __name__ == '__main__':
    main()
