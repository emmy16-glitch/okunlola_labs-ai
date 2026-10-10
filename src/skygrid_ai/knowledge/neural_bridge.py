"""Local Transformer draft beside (never in place of) audited source evidence.

The v0.9c/v1.0 models are next-token predictors, not instruction-tuned QA
models or fact-checkers. Only the existing extractive path supplies the
citation-backed excerpt. The neural continuation is a separate research field.
No model file, dataset, or source passage is modified here.
"""
from pathlib import Path

from .grounded import answer_from_evidence
from .retrieval import CORPUS, ROOT

VALID_MODELS = {'v09c', 'v10', 'qa_v15'}
VALID_CHECKPOINTS = {'last', 'best'}


def make_prompt(question, excerpt, tokenizer, context_size):
    """Fit question AND evidence into the actual token context, leaving room.

    Truncate at word boundaries so the byte BPE never creates an invalid
    partial Unicode prompt. This limits input context, not model output length.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Provide a nonempty question')
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError('Missing verified source excerpt')
    if context_size < 32:
        raise ValueError('Context is too small to fit question and evidence')
    budget = context_size - 1
    question_words = question.split()
    excerpt_words = excerpt.split()
    # The documentation excerpt has already passed the extractive source gate.
    # Preserve the question and a useful leading portion of the evidence.
    question_words = question_words[:80]
    excerpt_words = excerpt_words[:100]
    while True:
        prompt = 'Question: ' + ' '.join(question_words) + '\nEvidence: ' + ' '.join(excerpt_words) + '\nAnswer:'
        if len(tokenizer.encode(prompt)) <= budget:
            return prompt
        if len(excerpt_words) > 4:
            excerpt_words.pop()
        elif len(question_words) > 1:
            question_words.pop()
        else:
            raise ValueError('Context is too small to hold the source-backed prompt')


def load_local_model_assets(*, model_version='v09c', checkpoint_name='last',
                            root=ROOT):
    """Load only local, compatible SkyGrid checkpoints with safe torch.load."""
    if model_version not in VALID_MODELS or checkpoint_name not in VALID_CHECKPOINTS:
        raise ValueError('Choose model v09c/v10/qa_v15 and checkpoint best/last')
    import torch
    from src.skygrid_ai.language.large_bpe import LargeByteBPE
    from src.skygrid_ai.language.train_large_transformer import sha256_file

    root = Path(root)
    tokenizer_file = root / 'data' / 'aviation' / 'tokenized_v09b' / 'tokenizer.json'
    ckpt_file = root / 'checkpoints' / model_version / (checkpoint_name + '.pt')
    if not ckpt_file.is_file():
        raise FileNotFoundError(
            f'Missing {model_version} checkpoint: {ckpt_file}. '
            'Use the v09c step-5000 checkpoint or train the v10 model locally.'
        )
    tokenizer = LargeByteBPE.load(tokenizer_file)
    checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    expected_format = 'skygrid_qa_v15_1' if model_version == 'qa_v15' else 'skygrid_v09c_1'
    if checkpoint.get('format_version') != expected_format:
        raise ValueError('Unrecognized or mismatched SkyGrid checkpoint format')
    if model_version == 'qa_v15':
        from src.skygrid_ai.language.qa_pilot import PROTOCOL, BENCHMARKS, digest_files
        if checkpoint.get('qa_protocol_sha256') != digest_files((PROTOCOL, *BENCHMARKS)):
            raise ValueError('QA training dataset/protocol has changed')
        original_folder = root / 'checkpoints' / 'v10'
        expected_source = checkpoint.get('source_checkpoint_sha256')
        if not isinstance(expected_source, str) or not any(
            (original_folder / (name + '.pt')).is_file() and
            sha256_file(original_folder / (name + '.pt')) == expected_source
            for name in ('last', 'best')
        ):
            raise ValueError('Source v10 checkpoint fingerprint mismatch')
    if (checkpoint.get('vocab_size') != tokenizer.vocab_size or
            checkpoint.get('tokenizer_fingerprint') != tokenizer.fingerprint()):
        raise ValueError('Checkpoint tokenizer metadata mismatch')
    data_digests = checkpoint.get('data_digests')
    if not isinstance(data_digests, dict) or set(data_digests) != {'train', 'val'}:
        raise ValueError('Checkpoint lacks train/validation fingerprint metadata')
    for split, expected in data_digests.items():
        if sha256_file(tokenizer_file.parent / (split + '.bin')) != expected:
            raise ValueError(f'Checkpoint dataset mismatch: {split}')
    if not isinstance(checkpoint.get('model_config'), dict):
        raise ValueError('Checkpoint model configuration missing')
    return tokenizer, checkpoint


def answer_with_model(index, question, *, corpus=CORPUS, top_k=8,
                      model_version='v09c', checkpoint_name='last',
                      max_new_tokens=64, temperature=0.7, top_k_tokens=10,
                      seed=7, root=ROOT, model_loader=None, generator=None):
    """Return source excerpt + explicitly unverified neural continuation.

    Abstention or source mismatch MUST NOT call the model. A naive next-token
    continuation is never substituted for a missing citation-backed answer.
    """
    if model_version not in VALID_MODELS or checkpoint_name not in VALID_CHECKPOINTS:
        raise ValueError('Choose model v09c/v10/qa_v15 and checkpoint best/last')
    if (not 1 <= max_new_tokens <= 256 or not 0 < temperature <= 2.0 or
            not 1 <= top_k_tokens <= 768 or not isinstance(seed, int)):
        raise ValueError('Invalid sampling settings')
    evidence = answer_from_evidence(index, question, top_k=top_k, corpus=corpus)
    result = {
        'question': question,
        'method': 'offline_verified_source_extraction_plus_experimental_neural_continuation',
        'evidence_status': evidence['status'],
        'source_excerpt': evidence.get('answer'),
        'citations': evidence.get('citations', []),
        'neural_draft': None,
        'neural_draft_status': 'not_generated',
        'warning': ('The cited text is extracted from a verified local source file, '
                    'not an independently fact-checked claim. Neural text is '
                    'unverified next-token generation, NEVER flight guidance.'),
    }
    if evidence['status'] != 'excerpt_found':
        result['reason'] = evidence.get('reason', 'No adequate source evidence')
        if evidence.get('suggested_sources'):
            result['suggested_sources'] = evidence['suggested_sources']
        return result
    if model_loader is None:
        model_loader = load_local_model_assets
    tokenizer, checkpoint = model_loader(
        model_version=model_version, checkpoint_name=checkpoint_name, root=root)
    context_size = checkpoint['model_config']['context_size']
    prompt = make_prompt(question, evidence['answer'], tokenizer, context_size)
    if generator is None:
        from src.skygrid_ai.language.generate_large import generate
        generator = generate
    # The shared generator returns prompt + generated text. Never label the
    # continuation as a verified fact, even if it repeats source terminology.
    continuation = generator(
        prompt, tokenizer=tokenizer, checkpoint=checkpoint,
        max_new_tokens=max_new_tokens, temperature=temperature,
        top_k=top_k_tokens, seed=seed,
    )
    if not isinstance(continuation, str) or not continuation.startswith(prompt):
        raise ValueError('Generator response does not preserve the prompt')
    draft = continuation[len(prompt):].strip()
    result['model'] = {'version': model_version, 'checkpoint': checkpoint_name,
                       'training_step': checkpoint['step']}
    result['neural_draft_status'] = 'generated_unverified' if draft else 'empty_continuation'
    result['neural_draft'] = draft or None
    return result
