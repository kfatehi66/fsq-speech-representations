"""
CMU Phoneme Vocabulary for LibriSpeech.
39 phonemes + special tokens.
"""

# CMU ARPAbet phonemes (39 base phonemes)
CMU_PHONEMES = [
    # Vowels (15)
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants (24)
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH',
]

# Special tokens
SPECIAL_TOKENS = ['<pad>', '<blank>', '<unk>', '<sil>']

# Full vocabulary
PHONEME_VOCAB = SPECIAL_TOKENS + CMU_PHONEMES

# Create mappings
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
ID_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_VOCAB)}

# Vocabulary size
VOCAB_SIZE = len(PHONEME_VOCAB)  # 43 = 4 special + 39 phonemes

# Blank ID for CTC
BLANK_ID = PHONEME_TO_ID['<blank>']
PAD_ID = PHONEME_TO_ID['<pad>']
SIL_ID = PHONEME_TO_ID['<sil>']


def normalize_phoneme(phoneme: str) -> str:
    """Normalize phoneme label from TextGrid."""
    # Remove stress markers (0, 1, 2)
    phoneme = phoneme.rstrip('012')

    # Handle silence variants
    if phoneme in ('', 'sp', 'sil', 'SIL', 'spn', 'SPN'):
        return '<sil>'

    # Convert to uppercase
    phoneme = phoneme.upper()

    # Return if in vocabulary
    if phoneme in PHONEME_TO_ID:
        return phoneme

    return '<unk>'


def encode_phonemes(phonemes: list) -> list:
    """Convert list of phonemes to IDs."""
    return [PHONEME_TO_ID.get(normalize_phoneme(p), PHONEME_TO_ID['<unk>']) for p in phonemes]


def decode_phonemes(ids: list, remove_special: bool = True) -> list:
    """Convert list of IDs to phonemes."""
    phonemes = [ID_TO_PHONEME.get(i, '<unk>') for i in ids]
    if remove_special:
        phonemes = [p for p in phonemes if p not in SPECIAL_TOKENS]
    return phonemes


if __name__ == '__main__':
    print(f"Phoneme vocabulary size: {VOCAB_SIZE}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print(f"CMU Phonemes ({len(CMU_PHONEMES)}): {CMU_PHONEMES}")
    print(f"Blank ID: {BLANK_ID}")
    print(f"Silence ID: {SIL_ID}")
