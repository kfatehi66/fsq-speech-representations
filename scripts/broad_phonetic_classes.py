"""
Broad Phonetic Class mappings from:
Loweimi et al. "Phonetic Error Analysis Beyond Phone Error Rate"
IEEE/ACM TASLP, 2023

Tables I and II define mappings from 39 phonemes to broad classes.

Uses uppercase CMU ARPAbet to match our phoneme_vocab.py conventions.
"""

# Table I: 8-class Broad Phonetic Classes (uppercase CMU ARPAbet)
BPC_8_CLASS = {
    # Affricates (2 phones)
    'CH': 'affricate', 'JH': 'affricate',

    # Diphthongs (5 phones)
    'AW': 'diphthong', 'AY': 'diphthong', 'EY': 'diphthong',
    'OW': 'diphthong', 'OY': 'diphthong',

    # Fricatives (7 phones here; ZH is added below)
    'DH': 'fricative', 'F': 'fricative', 'S': 'fricative',
    'SH': 'fricative', 'TH': 'fricative', 'V': 'fricative', 'Z': 'fricative',

    # Nasals (3 phones)
    'M': 'nasal', 'N': 'nasal', 'NG': 'nasal',

    # Plosives (6 phones; DX is not in the 39-phoneme set and is not listed)
    'B': 'plosive', 'D': 'plosive', 'G': 'plosive',
    'K': 'plosive', 'P': 'plosive', 'T': 'plosive',

    # Semi-vowels (5 phones)
    'HH': 'semi-vowel', 'L': 'semi-vowel', 'R': 'semi-vowel',
    'W': 'semi-vowel', 'Y': 'semi-vowel',

    # Vowels (10 phones, including AO from the CMU set)
    'AA': 'vowel', 'AE': 'vowel', 'AH': 'vowel', 'AO': 'vowel',
    'EH': 'vowel', 'ER': 'vowel', 'IH': 'vowel', 'IY': 'vowel',
    'UH': 'vowel', 'UW': 'vowel',

    # Silence / special
    '<sil>': 'silence',

    # ZH is in our vocab but rare; treat as fricative
    'ZH': 'fricative',
}

# Table II: 3-class Consonant/Vowel+/Silence
BPC_3_CLASS_CVS = {
    # Vowel+ = Vowels + Diphthongs
    'AW': 'vowel+', 'AY': 'vowel+', 'EY': 'vowel+', 'OW': 'vowel+', 'OY': 'vowel+',
    'AA': 'vowel+', 'AE': 'vowel+', 'AH': 'vowel+', 'AO': 'vowel+',
    'EH': 'vowel+', 'ER': 'vowel+', 'IH': 'vowel+', 'IY': 'vowel+',
    'UH': 'vowel+', 'UW': 'vowel+',

    # Consonant
    'B': 'consonant', 'CH': 'consonant', 'D': 'consonant', 'DH': 'consonant',
    'F': 'consonant', 'G': 'consonant', 'HH': 'consonant',
    'JH': 'consonant', 'K': 'consonant', 'L': 'consonant', 'M': 'consonant',
    'N': 'consonant', 'NG': 'consonant', 'P': 'consonant', 'R': 'consonant',
    'S': 'consonant', 'SH': 'consonant', 'T': 'consonant', 'TH': 'consonant',
    'V': 'consonant', 'W': 'consonant', 'Y': 'consonant', 'Z': 'consonant',
    'ZH': 'consonant',

    # Silence
    '<sil>': 'silence',
}

# Table II: 3-class Voiced/Unvoiced/Silence
BPC_3_CLASS_VUS = {
    # Voiced
    'AA': 'voiced', 'AE': 'voiced', 'AH': 'voiced', 'AO': 'voiced',
    'AW': 'voiced', 'AY': 'voiced',
    'B': 'voiced', 'D': 'voiced', 'DH': 'voiced',
    'EH': 'voiced', 'ER': 'voiced', 'EY': 'voiced',
    'G': 'voiced', 'HH': 'voiced',
    'IH': 'voiced', 'IY': 'voiced',
    'JH': 'voiced', 'L': 'voiced', 'M': 'voiced', 'N': 'voiced', 'NG': 'voiced',
    'OW': 'voiced', 'OY': 'voiced',
    'R': 'voiced', 'UH': 'voiced', 'UW': 'voiced',
    'V': 'voiced', 'W': 'voiced', 'Y': 'voiced', 'Z': 'voiced',
    'ZH': 'voiced',

    # Unvoiced
    'CH': 'unvoiced', 'F': 'unvoiced', 'K': 'unvoiced', 'P': 'unvoiced',
    'S': 'unvoiced', 'SH': 'unvoiced', 'T': 'unvoiced', 'TH': 'unvoiced',

    # Silence
    '<sil>': 'silence',
}

# Class to index mappings
BPC_8_TO_IDX = {
    'affricate': 0, 'diphthong': 1, 'fricative': 2, 'nasal': 3,
    'plosive': 4, 'semi-vowel': 5, 'vowel': 6, 'silence': 7
}

BPC_3_CVS_TO_IDX = {'consonant': 0, 'vowel+': 1, 'silence': 2}

BPC_3_VUS_TO_IDX = {'voiced': 0, 'unvoiced': 1, 'silence': 2}

# Index to class name (for interpretability)
IDX_TO_BPC_8 = {v: k for k, v in BPC_8_TO_IDX.items()}
IDX_TO_BPC_3_CVS = {v: k for k, v in BPC_3_CVS_TO_IDX.items()}
IDX_TO_BPC_3_VUS = {v: k for k, v in BPC_3_VUS_TO_IDX.items()}


def phoneme_to_bpc8_idx(phoneme: str) -> int:
    """Convert uppercase phoneme to 8-class BPC index. Returns -1 if unknown."""
    bpc = BPC_8_CLASS.get(phoneme)
    if bpc is None:
        return -1
    return BPC_8_TO_IDX[bpc]


def phoneme_to_bpc3_cvs_idx(phoneme: str) -> int:
    """Convert uppercase phoneme to 3-class (Consonant/Vowel+/Silence) index."""
    bpc = BPC_3_CLASS_CVS.get(phoneme)
    if bpc is None:
        return -1
    return BPC_3_CVS_TO_IDX[bpc]


def phoneme_to_bpc3_vus_idx(phoneme: str) -> int:
    """Convert uppercase phoneme to 3-class (Voiced/Unvoiced/Silence) index."""
    bpc = BPC_3_CLASS_VUS.get(phoneme)
    if bpc is None:
        return -1
    return BPC_3_VUS_TO_IDX[bpc]


def phoneme_id_to_bpc8_id(phoneme_id: int) -> int:
    """Convert phoneme_vocab ID directly to BPC-8 ID."""
    from data.phoneme_vocab import ID_TO_PHONEME
    phoneme = ID_TO_PHONEME.get(phoneme_id, '<unk>')
    return phoneme_to_bpc8_idx(phoneme)


# Build lookup table: phoneme_vocab ID -> BPC-8 ID
def build_phoneme_to_bpc_table(bpc_type='8class'):
    """Build a lookup table from phoneme_vocab IDs to BPC IDs.

    Returns a list where index = phoneme_vocab ID, value = BPC class ID.
    Unknown/special tokens map to -100 (ignore_index for cross-entropy).
    """
    from data.phoneme_vocab import PHONEME_VOCAB

    if bpc_type == '8class':
        mapping = BPC_8_CLASS
        class_to_idx = BPC_8_TO_IDX
    elif bpc_type == '3class_cvs':
        mapping = BPC_3_CLASS_CVS
        class_to_idx = BPC_3_CVS_TO_IDX
    elif bpc_type == '3class_vus':
        mapping = BPC_3_CLASS_VUS
        class_to_idx = BPC_3_VUS_TO_IDX
    else:
        raise ValueError(f"Unknown BPC type: {bpc_type}")

    table = []
    for phoneme in PHONEME_VOCAB:
        bpc_class = mapping.get(phoneme)
        if bpc_class is not None:
            table.append(class_to_idx[bpc_class])
        else:
            table.append(-100)  # ignore

    return table


def verify_mappings():
    from data.phoneme_vocab import CMU_PHONEMES

    missing_8 = [p for p in CMU_PHONEMES if p not in BPC_8_CLASS]
    missing_cvs = [p for p in CMU_PHONEMES if p not in BPC_3_CLASS_CVS]
    missing_vus = [p for p in CMU_PHONEMES if p not in BPC_3_CLASS_VUS]

    print(f"CMU Phonemes: {len(CMU_PHONEMES)}")

    if missing_8:
        print(f"Warning: Missing from BPC_8_CLASS: {missing_8}")
    else:
        print(f"8-class: all {len(CMU_PHONEMES)} CMU phonemes mapped -> {len(BPC_8_TO_IDX)} classes")

    if missing_cvs:
        print(f"Warning: Missing from BPC_3_CLASS_CVS: {missing_cvs}")
    else:
        print(f"3-class CVS: all {len(CMU_PHONEMES)} CMU phonemes mapped -> {len(BPC_3_CVS_TO_IDX)} classes")

    if missing_vus:
        print(f"Warning: Missing from BPC_3_CLASS_VUS: {missing_vus}")
    else:
        print(f"3-class VUS: all {len(CMU_PHONEMES)} CMU phonemes mapped -> {len(BPC_3_VUS_TO_IDX)} classes")

    # Print class distributions
    print("\n8-class distribution:")
    for cls_name, cls_idx in sorted(BPC_8_TO_IDX.items(), key=lambda x: x[1]):
        members = [p for p, c in BPC_8_CLASS.items() if c == cls_name and p != '<sil>']
        print(f"  [{cls_idx}] {cls_name:12s}: {members}")

    # Test the lookup table
    table = build_phoneme_to_bpc_table('8class')
    print(f"\nPhoneme->BPC lookup table ({len(table)} entries):")
    from data.phoneme_vocab import PHONEME_VOCAB
    for i, (p, b) in enumerate(zip(PHONEME_VOCAB, table)):
        bpc_name = IDX_TO_BPC_8.get(b, 'IGNORE') if b >= 0 else 'IGNORE'
        print(f"  {i:2d} {p:6s} -> {b:4d} ({bpc_name})")


if __name__ == '__main__':
    verify_mappings()
